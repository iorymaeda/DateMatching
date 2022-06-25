import os
import pickle
import datetime

import torch
import torchmetrics
import torchsummary
import numpy as np
import torch.nn.functional as F
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split







if __name__ == "__main__":
    import os
    import sys
    import inspect

    # Import from parent directory
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 
    import utils

    embeder = utils.PhotoEmbedingStorage('emb storage.pkl')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBED_DIM = 32

    files_opposite = []
    files_target = []

    path = "data/markup_opposite/"
    files_opposite += [int(f.split('.')[0]) for f in os.listdir(path)]
        
    path =  "data/markup_target/"
    files_target += [int(f.split('.')[0]) for f in os.listdir(path)]

    path = "data/opposite/"
    files_opposite += [int(f.split('.')[0]) for f in os.listdir(path)]

    path =  "data/target/"
    files_target += [int(f.split('.')[0]) for f in os.listdir(path)]

    files_opposite = np.array(files_opposite)
    files_target = np.array(files_target)

    y_opposite = np.zeros_like(files_opposite, dtype='float32')
    y_target = np.ones_like(files_target, dtype='float32')

    X = np.concatenate([files_opposite, files_target])
    Y = np.concatenate([y_opposite, y_target])

    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.2, random_state=69)

    Xtest = []
    Ytest = []

    path = "data/test_target/"
    ldir = [int(f.split('.')[0]) for f in os.listdir(path)]
    Xtest+= ldir
    Ytest+= [1. for _ in ldir]
        
    path = "data/test_opposite/"
    ldir = [int(f.split('.')[0]) for f in os.listdir(path)]
    Xtest+= ldir
    Ytest+= [0. for _ in ldir]

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest, dtype='float32')

    embeds = np.concatenate([embeder[int(xid)] for xid in Xtrain])


    pca = PCA(n_components=EMBED_DIM)
    pca.fit(embeds)

    pickle.dump(pca, open("../Models/pca.pkl","wb"))

    trainLoader = torch.utils.data.DataLoader(
        utils.EmbedDataset(
            x=Xtrain, 
            y=Ytrain, 
            embeder=embeder,
            decompositor=pca.transform), 
        batch_size=2048, 
    )

    valLoader = torch.utils.data.DataLoader(
        utils.TestDataset(
            x=Xval, 
            y=Yval, 
            embeder=embeder,
            decompositor=pca.transform), 
        batch_size=2048, 
    )

    testLoader = torch.utils.data.DataLoader(
        utils.TestDataset(
            x=Xtest, 
            y=Ytest, 
            embeder=embeder,
            decompositor=pca.transform), 
        batch_size=2048, 
    )

    model = utils.Model(EMBED_DIM, d=[64, 64])
    trainer = utils.Trainer(
        model=model.cuda(),
        stop_batch=10_000/2048,
        metric=torchmetrics.AUROC(),
        loss_fn=nn.BCEWithLogitsLoss(reduce=True),
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
    )

    acc = torchmetrics.Accuracy()
    auc = torchmetrics.AUROC()

    torchsummary.summary(model)


    name = 'InceptionResnetV1 vggface2 pca 32 '
    board_name = name + datetime.datetime.now().strftime("%Y.%m.%d - %H-%M-%S")

    log_dir = f"logs/fit/{board_name}"
    writer = SummaryWriter(log_dir)


    try:
        wait = 0
        patience = 50
        
        epoch = 0
        best_loss = -np.inf
        while wait < patience:
            train_loss = trainer.train(trainLoader, epoch)

            val_pred, val_true = trainer.val(valLoader)
            metrics = {
                'AUC': auc(val_pred.sigmoid(), val_true.int()),
                'ACC': acc(val_pred.sigmoid(), val_true.int()),
            }
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('AUC/train', trainer.metric.compute(), epoch)
            writer.add_scalar('AUC/val', metrics['AUC'], epoch)
            writer.add_scalar('ACC/val', metrics['ACC'], epoch)


            wait+=1
            epoch+=1
            if metrics['AUC'] > best_loss:
                checkpoint = trainer.checkpoint()
                torch.save(checkpoint, f'../Models/w/{name}.torch')
                best_loss = metrics['AUC']
                wait = 0


    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    checkpoint = torch.load(f'../Models/w/{name}.torch')
    model.load_state_dict(checkpoint['model'])

    test_pred, test_true = trainer.val(testLoader)
    print('AUC:', auc(test_pred.sigmoid(), test_true.int()))
    print('ACC:', acc(test_pred.sigmoid(), test_true.int()))
    torch.save(checkpoint, f'../Models/w/prod.torch')