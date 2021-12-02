import torch
import os
from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification, BertConfig
from torch import nn
import numpy as np
from transformers import xnli_compute_metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import logging
from training import test_config
from sentence_transformers.util import batch_to_device, dump_json

logger = logging.getLogger(__name__)


class NLIBert(nn.Module):

    def __init__(self, checkpoint, label_map, device, num_labels):
        super(NLIBert, self).__init__()

        # load model
        # small bert is a toy model for debugging
        if checkpoint == 'small_bert':
            test_config.test_config['num_labels'] = num_labels
            bert_config = BertConfig.from_dict(test_config.test_config)
            self.model = BertForSequenceClassification(bert_config)
        elif checkpoint == 'xlm-roberta-base':
            self.model = XLMRobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint, num_labels=len(label_map))
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint, num_labels=len(label_map))
        logger.info('Using {} architecture'.format(self.model.name_or_path))
        self.model.to(device)
        self.device = device

        headline = '############# Model Arch of MT-DNN #############'
        logger.info('\n{}\n{}\n'.format(headline, self.model))
        logger.info("Total number of params: {}".format(sum([p.nelement() for p in self.model.parameters() if p.requires_grad])))

        self.label2id = label_map
        self.id2label = {val:key for key, val in self.label2id.items()}


    def fit(self, optimizer, scheduler, train_dataloader, dev_dataloader, test_dataloader, epochs, grad_accumulation_steps, evaluation_step, save_best, outdir, predict):

        # get lr schedule
        total_steps = (len(train_dataloader)/grad_accumulation_steps) * epochs

        loss_values = []
        global_step = 0

        best_dev_score = 0
        epoch = -1
        for epoch in range(epochs):
            logger.info('Starting epoch {}'.format(epoch))

            total_loss = 0
            accumulated_steps = 0
            steps_trained_in_current_epoch = 0


            # clear gradients
            self.model.zero_grad()

            for ste, batch in enumerate(train_dataloader):

                self.model.train()

                # batch to device
                batch.to(self.device)

                # perform forward pass
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

                # compute loss
                loss = output[0]

                if grad_accumulation_steps > 1:
                    loss = loss / grad_accumulation_steps

                # perform backward pass
                loss.backward()

                total_loss += loss.item()

                # gradient gets accumulated. if grad_acc step size is reached, perform a backward pass
                accumulated_steps += 1

                if accumulated_steps%grad_accumulation_steps == 0:

                    # clip the gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Update parameters and take a step using the computed gradient.
                    # The optimizer dictates the "update rule"--how the parameters are
                    # modified based on their gradients, the learning rate, etc.

                    # take a step and update the model
                    optimizer.step()

                    # Update the learning rate.
                    scheduler.step()
                    optimizer.zero_grad()

                    # clear gradients
                    self.model.zero_grad()

                    global_step += 1
                    steps_trained_in_current_epoch += 1




                    # evaluate on dev
                    if steps_trained_in_current_epoch > 0 and steps_trained_in_current_epoch % evaluation_step == 0:
                        self.model.eval()
                        dev_score, dev_results, _ = self.evaluate_on_dev(data_loader=dev_dataloader)
                        logger.info('Epoch {}, global step {}/{}\ttrain loss: {:.5f}\t dev score: {}'.format(epoch,
                                                                                                             global_step,
                                                                                                             total_steps, total_loss/steps_trained_in_current_epoch,
                                                                                                            dev_score))


            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / steps_trained_in_current_epoch

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            # evaluate on dev after epoch is finished
            self.model.eval()

            dev_score, dev_results, dev_predictions = self.evaluate_on_dev(data_loader=dev_dataloader)

            logger.info('End of epoch {}, global step {}/{}\ttrain loss: {:.5f}\t dev score: {:.5f}\ndev report {}'.format(epoch,
                                                                                                     global_step,
                                                                                                     total_steps,
                                                                                                     total_loss / steps_trained_in_current_epoch,
                                                                                                     dev_score, dev_results))
            dump_json(fname=os.path.join(outdir, 'dev_preds_{}.json'.format(epoch)),
                      data={'score': dev_score, 'results': dev_results, 'predictions': dev_predictions})
            if dev_score >= best_dev_score:
                logger.info('New dev score {:.5f} > {:.5f}'.format(dev_score, best_dev_score))
                best_dev_score = dev_score
                if save_best:
                    #save model
                    logger.info('Saving model after epoch {} as best model to {}'.format(epoch, os.path.join(outdir, 'best_model')))
                    self.save(os.path.join(outdir, 'best_model/model_{}.pt'.format(epoch)))

                    if predict:
                        logger.info('Predicting test data with best model at end of epoch {}'.format(epoch))
                        self.model.eval()
                        test_score, test_results, test_predictions = self.evaluate_on_dev(data_loader=test_dataloader)
                        # dump to file
                        dump_json(fname=os.path.join(outdir, 'best_model/test_preds_{}.json'.format(epoch)),
                                  data={'score': test_score, 'results': test_results, 'predictions': test_predictions})




            if predict:
                logger.info('Predicting test data at end of epoch {}'.format(epoch))
                self.model.eval()
                test_score, test_report, test_predictions = self.evaluate_on_dev(data_loader=test_dataloader)
                # dump to file
                dump_json(fname=os.path.join(outdir, 'test_preds_{}.json'.format(epoch)),
                          data={'score': test_score, 'results': test_results, 'predictions': test_predictions})



        if not save_best:
            # save model
            logger.info('Saving model after epoch {} to {}'.format(epoch, os.path.join(outdir)))
            self.save(os.path.join(outdir, 'model_{}.pt'.format(epoch)))




    def evaluate_on_dev(self, data_loader):

        self.model.eval()
        preds = None
        out_label_ids = None

        for step, batch in enumerate(data_loader):
            # batch to device
            batch.to(self.device)

            # perform forward pass
            with torch.no_grad():
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss, logits = output[:2]
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = batch["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)

        results = self.compute_metrics(golds=out_label_ids, preds=preds, label_map=self.id2label)
        return results['macro avg']['f1-score'], results, preds.tolist()



    def compute_metrics(self, golds, preds, label_map):

        report = classification_report(y_true=golds, y_pred=preds, labels=[0,1,2], target_names=[label_map[key] for key in [0,1,2]], sample_weight=None,
                                               output_dict=True, zero_division='warn')

        return  report





    def save(self, outpath):
        outpath = '/'.join(outpath.split('/')[:-1])
        self.model.save_pretrained(outpath)



