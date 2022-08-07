import pandas as pd
import torch

from torch.utils import data
import numpy as np
from sklearn.metrics import r2_score, classification_report, roc_auc_score, average_precision_score
import datetime


OUTPUT_DIR = 'output/'
TENSORBOARDX_OUTPUT_DIR = 'tbxoutput/'
SAVEDMODELS_DIR = 'savedmodels/'
# time of importing this file, including microseconds because slurm may start queued jobs very close in time
DATETIME_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


class Globals: # container for all objects getting passed between log calls
    evaluate_called = False

g = Globals()

TRAIN_SUBSET_SIZE = 500
SUBSET_LOADER_BATCH_SIZE = 50



def all_evaluate(output,target):

    lengh=len(output)
    scores = torch.sigmoid(output)
    #print('scores:',scores)
    #print('target:',target)

    auroc = roc_auc_score(target, scores)
    prauc = average_precision_score(target, scores)
    scores = np.array(scores).astype(float)

    sum_scores = np.sum(scores)
    ave_scores = sum_scores / lengh

    target = np.array(target).astype(int)

    Confusion_M = np.zeros((2, 2), dtype=float)  # (TN FP),(FN,TP)
    for i in range(lengh):
        if (scores[i] < ave_scores):
            scores[i] = 0
        else:
            scores[i] = 1

    scores = np.array(scores).astype(int)

    for i in range(lengh):
        #print(i)
        #print('target[i],scores[i]:',target[i],scores[i])
        if(target[i]==scores[i]):
            if(target[i]==1):
                Confusion_M[0][0] += 1#TP
            else:
                Confusion_M[1][1] += 1#TN
        else:
            if(target[i]==1):
                Confusion_M[0][1] += 1#FP
            else:
                Confusion_M[1][0]  +=1#FN


    Confusion_M = np.array(Confusion_M, dtype=float)
    print('Confusion_M:', Confusion_M)
    #M[TP,FN
    #  FP,TN]

    accuracy = (Confusion_M[1][1] + Confusion_M[0][0]) / (
            Confusion_M[0][0] + Confusion_M[1][1] + Confusion_M[0][1] + Confusion_M[1][0])

    recall = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])  # 分类正确的正样本，占预测为正的比例
    #recall=TP/TP+FN
    #precision=TP/TP+FP
    precision = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[1][0])  # 分类正确的正样本占预测为正的正样本的比例
    F1=2*precision*recall
    h=precision+recall
    F1=F1/h

    # True Positive Rate: TPR = TP / (TP + FN) → 将正例分对的概率
    # Fales Positive Rate: FPR = FP / (FP + TN) → 将负例错分为正例的概率
    TPR = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])
    #TP/TP+FN
    FPR = Confusion_M[0][1] / (Confusion_M[1][0] + Confusion_M[1][1])
    #FP/TN+FP



    return accuracy,auroc,prauc,F1,recall,precision,


SCORE_FUNCTIONS = {
    'All':all_evaluate
}


def feed_net(net,dataloader, criterion, cuda):
    #print('len(dataloader)',len(dataloader))
    batch_outputs = []
    batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):
        if cuda:
            batch = [tensor.cuda(non_blocking=True) for tensor in batch]
        #print('batch:',len(batch))
        adjacency_1, nodes_1, edges_1, adjacency_2, nodes_2, edges_2, d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts, \
        target, side_effect, d1, d2 = batch
        #print('batch target:',target)
        output = net(adjacency_1, nodes_1, edges_1,adjacency_2, nodes_2, edges_2,d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts,side_effect)



        loss = criterion(output, target)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(target)

    outputs = torch.cat(batch_outputs)
    #print('loss',batch_losses)
    loss = np.mean(batch_losses)#平均loss
    targets = torch.cat(batch_targets)
    return outputs, loss, targets




def evaluate_net(net,train_dataloader, validation_dataloader, test_dataloader, criterion, args):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score = 0, 0, 0

        g.train_subset_loader = train_dataloader


    train_output, train_loss, train_target = feed_net(net,g.train_subset_loader, criterion, args.cuda)
    validation_output, validation_loss, validation_target = feed_net(net,validation_dataloader, criterion, args.cuda)
    test_output, test_loss, test_target = feed_net(net,test_dataloader, criterion, args.cuda)
    train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)

    new_best_model_found = validation_scores[1] > g.best_mean_validation_score  #auc


    if new_best_model_found:
        g.best_mean_train_score = train_scores[1]
        g.best_mean_validation_score = validation_scores[1]
        g.best_mean_test_score = test_scores[1]

        if args.savemodel:
            #path = SAVEDMODELS_DIR + type(net).__name__ + DATETIME_STR
            path = SAVEDMODELS_DIR + type(net).__name__
            torch.save(net, path)

            t_score=torch.sigmoid(test_output).numpy()
            test_pred = pd.DataFrame(data=t_score)
            test_pred.to_csv('./output/test_pred.csv', index=False)

            t_target=test_target.numpy()
            test_label = pd.DataFrame(data=t_target)
            test_label.to_csv('./output/test_label.csv',index=False)

    #accuracy, auroc, prauc, F1, recall, precision,

    if(args.score=='All'):
     return{
         'loss':{'train': train_loss, 'test': test_loss},
         'F1 score':{'train': train_scores[3], 'validation': validation_scores[3], 'test': test_scores[3]},
         'Accuracy':{'train': train_scores[0], 'validation': validation_scores[0], 'test': test_scores[0]},
         'Recall':{'train': train_scores[4], 'validation': validation_scores[4], 'test': test_scores[4]},
         'Precision':{'train': train_scores[5], 'validation': validation_scores[5], 'test': test_scores[5]},
         'auroc': {'train': train_scores[1], 'validation': validation_scores[1], 'test': test_scores[1]},
         'prauc': {'train': train_scores[2], 'validation': validation_scores[2], 'test': test_scores[2]},


         'best mean':{'train': g.best_mean_train_score, 'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
        }




def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in  net._modules.items()}
    }


def less_log(net,train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):

    scalars = evaluate_net(net,train_dataloader, validation_dataloader, test_dataloader, criterion, args)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')


    #返回的包含四种分数，和最优结果
    if(args.score=='All'):
        print('epoch {}'.format(epoch + 1))

        print('          AUROC:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['auroc']['train'],
            scalars['auroc']['validation'],
            scalars['auroc']['test']))
        print('          PRAUC:training mean: {}, testing mean: {}'.format(
            scalars['prauc']['train'],
            scalars['prauc']['test']))

        print('          ACC:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['Accuracy']['train'],
            scalars['Accuracy']['validation'],
            scalars['Accuracy']['test']))

        print('         F1 score :training mean: {}, validation mean: {}, testing mean: {}'.format(
         scalars['F1 score']['train'],
         scalars['F1 score']['validation'],
            scalars['F1 score']['test'])
            )
        print('          Precision:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['Precision']['train'],
            scalars['Precision']['validation'],
            scalars['Precision']['test']))
        print('          Recall:training mean: {}, validation mean: {}, testing mean: {}'.format(
            scalars['Recall']['train'],
            scalars['Recall']['validation'],
            scalars['Recall']['test']))

        print('          best auc:training mean: {}, validation mean: {}, testing mean: {}'.format(
             scalars['best mean']['train'],
             scalars['best mean']['validation'],
                scalars['best mean']['test']))

        print('          loss:training mean: {},testing mean: {}'.format(
            scalars['loss']['train'],
            scalars['loss']['test']))

    else:
         mean_score_key = 'mean {}'.format(args.score)
         print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}:{}'.format(
                epoch + 1,
                args.score, scalars[mean_score_key]['train'],
                args.score, scalars[mean_score_key]['validation'],
                args.score, scalars[mean_score_key]['test']),
         )





def more_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    mean_score_key = 'mean {}'.format(args.score)
    best_mean_score_key = 'best {}'.format(mean_score_key)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')


        print('score metric: {}'.format(args.score))
        print('columns:')
        print(
            'epochs, ' + \
            'mean training score, mean validation score, mean testing score, ' + \
            'best-model-so-far mean training score, best-model-so-far mean validation score, best-model-so-far mean testing score'
        )

    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args)
    print(
        '%d, %f, %f, %f, %f, %f, %f' % (
            epoch + 1,
            scalars[mean_score_key]['train'], scalars[mean_score_key]['validation'], scalars[mean_score_key]['test'],
            scalars[best_mean_score_key]['train'], scalars[best_mean_score_key]['validation'], scalars[best_mean_score_key]['test']
        )
    )


LOG_FUNCTIONS = {
    'less': less_log, 'more': more_log
}

