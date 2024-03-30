import torch
import time
import os
import numpy as np
from torch.sparse import *
os.environ['DGLBACKEND'] = 'pytorch'
from scipy import stats
from copy import deepcopy
from torch import optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import pickle

test_acc_lb, test_acc_lb_var = 82.5, 0.7

def plot_losses(train_losses, val_losses, modelname, log=False):
    """
    Plots train/validation loss curves vs training epoch
    """
    fig, ax = plt.subplots()

    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set(xlabel='Epoch', ylabel='CrossEnt')
    if log:
        ax.set_yscale('log')
    ax.legend()
    ax.grid()
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"plot_losses_{timestamp,modelname}.png"
    picklename_t = f"train_losses_{timestamp,modelname}.pkl"
    picklename_l = f"train_losses_{timestamp,modelname}.pkl"
    plt.savefig(filename)
    
    with open(picklename_t, 'wb') as file:
        pickle.dump(train_losses, file)
    
    with open(picklename_l, 'wb') as file:
        pickle.dump(val_losses, file)
        
    
def train(graph, labels, model, epochs, 
          device, save_path, loss_fn=F.cross_entropy, lr=0.01, es_criteria=5, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0
    train_losses = list()
    val_losses = list()
    
    labels = labels
    features = graph.ndata['feat']
    #the masks are converted to torch.uint8 when loaded from file. Better to have them in bool.
    train_mask = graph.ndata['train_mask'].bool().to(device)
    val_mask = graph.ndata['val_mask'].bool().to(device)
    test_mask = graph.ndata['test_mask'].bool().to(device)
    es_iters = 0

    for e in range(1, epochs+1):
        
        train_loss, val_loss = train_step(
            model, graph, features, labels, train_mask, val_mask, optimizer, loss_fn
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Compute accuracy on training/validation/test
        train_acc, val_acc, test_acc = test(model, graph, labels, train_mask, val_mask, test_mask)

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            es_iters = 0
        else:
            es_iters += 1
            

        if e % 50 == 0 and verbose:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, train_loss, val_acc, best_val_acc, test_acc, best_test_acc))
            
        if es_iters >= es_criteria:
            print(f"Early stopping at {e} epochs")
            break
            
    return np.array(train_losses), np.array(val_losses)

def train_step(model, graph, features, labels, train_mask, val_mask, optimizer, loss_fn):
    """
    A single training step
    """
    model.train()

    optimizer.zero_grad()
    logits = model(graph, features)
    loss = loss_fn(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        val_loss = loss_fn(logits[val_mask], labels[val_mask])

    return loss.item(), val_loss.item()

@torch.no_grad()
def test(model_cp, graph, labels, train_mask, val_mask, test_mask, best_path=None):
    model = deepcopy(model_cp)
    
    if best_path is not None:
        model.load_state_dict(torch.load(best_path))
        
    model.eval()
    
    features = graph.ndata['feat']
    logits = model(graph, features)
    y_pred = logits.argmax(1, keepdim=True)
    #y_pred = logits.argmax(1)

    train_acc = (y_pred[train_mask] == labels[train_mask]).float().mean()
    val_acc = (y_pred[val_mask] == labels[val_mask]).float().mean()
    test_acc = (y_pred[test_mask] == labels[test_mask]).float().mean()

    return train_acc, val_acc, test_acc

def characterize_performance(model, graph, labels, train_mask, val_mask, test_mask, best_path, verbose=False):
    train_acc, val_acc, test_acc = test(model, graph, labels, train_mask, val_mask, test_mask, best_path)
    print(
        f"Leaderboard:  Test Acc={test_acc_lb} +/- {test_acc_lb_var}\n"
        f"Yours:        Test Acc={test_acc:.4f},            Val Acc={val_acc:.4f}\n"
    )
    
    test_lb = test_acc_lb - test_acc_lb_var
    test_ub = test_acc_lb + test_acc_lb_var
    if verbose:
        if not test_acc >= test_lb:
            print(
                f"Test performance is worse than LB.  Expected lower bound of {test_lb:.4f}, but got {test_acc:.4f}.")

        elif test_acc > test_ub:
            print(
                f"Test performance is better than LB.  Expected upper bound of {test_ub:.4f}, but got {test_acc:.4f}.")
        else:
            print(f"Test performance is in the expected range of {test_lb} - {test_ub}.")
        
    return val_acc, test_acc

def norm_plot(curves, title):
    fig, ax = plt.subplots()
    for mu, sigma, label in curves:
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label=label)
    
    ax.set_title(title)
    ax.legend()
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"norm_plot_{timestamp}.png"
    plt.savefig(filename)
    
def get_experiment_stats(model_cls, model_args, train_args, n_experiments=10):
    print(train_args)
    results = dict()
    for i in range(n_experiments):
        model = model_cls(**model_args).to(train_args['device'])
        print(f"Starting training for experiment {i+1}")
        # Add experiment number to model save_path
        train_args_cp = deepcopy(train_args)
        save_path, file_ext = train_args_cp.pop('save_path').split('.')
        timestamp = time.strftime("%Y%m%d%H%M%S")
        save_path_mod = f"{save_path}__{timestamp}_{i}.{file_ext}"
        
        train_losses, val_losses = train(model=model, save_path=save_path_mod, **train_args_cp)
        val_acc, test_acc = characterize_performance(
            model, train_args['graph'], train_args['labels'], train_args['split_idx'], 
            train_args['evaluator'], save_path_mod, train_args.get('verbose', False))
        
        results[i] = dict(val_acc=val_acc, test_acc=test_acc)
        print("Training complete\n")
        
    df_stats = pd.DataFrame(results).T.agg(['mean', 'std'])
    return df_stats
