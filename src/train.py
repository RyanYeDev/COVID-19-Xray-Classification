from torch.utils.tensorboard import SummaryWriter

from model import *
from data_loader import  *

# model
covid_cnn = Covid_cnn().cuda()

# loss function
loss_fn = nn.BCEWithLogitsLoss().cuda()

# optimizer
learning_rate = 1e-4
optimizer = torch.optim.SGD(covid_cnn.parameters(), learning_rate)

# count train times
total_train_step = 0
# count test times
total_test_step = 0
# count epoch times
epoch = 100

# tensorboard
writer= SummaryWriter("../logs")

for i in range(epoch):
    print("-------start {} train-------".format(i + 1))
    # start train
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.float().unsqueeze(1).cuda()
        outputs = covid_cnn(imgs)
        loss = loss_fn(outputs, targets)
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train_times：{}, Loss：{}". format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # start test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.float().unsqueeze(1).cuda()
            outputs = covid_cnn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            preds = torch.sigmoid(outputs) >= 0.5
            accuracy = (preds.int() == targets.int()).sum().item()
            total_accuracy += accuracy

    print("total loss: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(covid_cnn.state_dict(), "../models/covid_{}.pth".format(i))
    print("model saved")

writer.close()