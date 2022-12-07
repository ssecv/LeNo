import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from model_ours import GateNet
from torch.backends import cudnn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

cudnn.benchmark = True
torch.manual_seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #

ckpt_path = './noise-abalation/' #
exp_name = ''
args = {
    'iter_num': 26380,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    # transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True,drop_last=True)

criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_mae = nn.L1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

def main():
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = GateNet()
    model_dict = model.state_dict()
    pretrained_dict=torch.load('./weight/100000_res50.pth')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    net = model.cuda().train()

    model_sigma_maps = []
    model_sigma_maps.append(model.cn1_sigma_map)
    model_sigma_maps.append(model.sigma_map_2)
    # for ablation study
    # model_sigma_maps.append(model.layer1[-1].sigma_map)
    # model_sigma_maps.append(model.layer1[-1].sigma_map_2)
    # layers = [model.layer1,model.layer2,model.layer3,model.layer4]
    # for layer in layers:
    #     for block in layer:
    #         model_sigma_maps.append(block.sigma_map)

    optimizer2 = torch.optim.SGD(model_sigma_maps,lr=1e-2,momentum=args['momentum'], weight_decay=0,nesterov=True)
    optimizer = optim.SGD([
        {'params': [param for name, param  in net.named_parameters() if name[-4:] == 'bias' and 'sigma' not in name],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'sigma' not in name],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    # for training phase 2, freeze
    # for p in model.conv1.parameters():
    #     p.requires_grad=False
    # for p in model.bn1.parameters(): # pytorch just freeze bn.weight and bn.bias，running_mean and running_var left
    #     p.requires_grad=False
    # model.bn1.eval()
    # model.sigma_map.requires_grad=False
    # model.sigma_map_2.requires_grad=False

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'optim_'+args['snapshot'] + '.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    open(log_path, 'w').write(str(args) + '\n\n')

    # for training phase 1
    train(net, optimizer2, optimizer, model_sigma_maps)
    # for training phase2
    # train(net, optimizer)

def train(net, optimizer2, optimizer,model_sigma_maps):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record, loss3_record, loss4_record, loss5_record, loss6_record, loss7_record, loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']

            epoch=curr_iter//1319
            harmonic = 0.0
            for i in range(1, epoch  + 2):
                harmonic += 1 / i

            coef = 1e-4 / harmonic
            sigma_map1 = model_sigma_maps[0]
            reg_term = -coef * torch.sqrt(sigma_map1)
            loss2 = reg_term.sum()
            for sigma_map in model_sigma_maps[1:]:
                reg_term = -coef * torch.sqrt(sigma_map)
                reg_loss = reg_term.sum()
                loss2 += reg_loss

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            inputs, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            output_fpn, output_final,noise_gt,noise_pre,side = net(inputs)
            loss1 = criterion(output_fpn, labels)
            loss2 = criterion(output_final, labels)
            total_loss = loss1+loss2
            for s in side:
                total_loss+=criterion(s,labels)

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)

            for sigma_map in model_sigma_maps:
                sigma_map.data = torch.clamp(sigma_map.data, 0.01)

            #############log###############
            curr_iter += 1
            log = '[epoch %d] [iter %d], [total loss %.5f],[loss1 %.5f],[loss1 %.5f],[lr %.13f] ' % \
                  (epoch, curr_iter, total_loss_record.avg, loss1_record.avg, loss2_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                print('noise pre loss : mse')
                torch.save(net.state_dict(), ckpt_path+'%d.pth'%curr_iter)
                torch.save(optimizer.state_dict(),ckpt_path+'optim_%d.pth'%curr_iter)
                return

def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record, loss3_record, loss4_record, loss5_record, loss6_record, loss7_record, loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        lossnoise_record=AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']

            epoch=curr_iter//1319

            inputs, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            output_fpn, output_final,noise_gt,noise_pre,side = net(inputs)
            output_fpn,output_final=net(inputs)
            loss1 = criterion(output_fpn, labels)
            loss2 = criterion(output_final, labels) # label=[0,1]
            total_loss = loss1+loss2
            for s in side:
                total_loss+=criterion(s,labels)

            noiseloss = criterion_mse(noise_pre,noise_gt)
            noiseloss += 0.01*criterion(noise_pre,noise_gt)
            total_loss+=noiseloss

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            lossnoise_record.update(noiseloss.item(),batch_size)

            curr_iter += 1
            log = '[epoch %d] [iter %d], [total loss %.5f],[loss1 %.5f],[noiseloss %.5f],[lr %.13f] ' % (epoch, curr_iter, total_loss_record.avg, loss1_record.avg, lossnoise_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                print('noise pre loss : mse')
                torch.save(net.state_dict(), ckpt_path+'%d.pth'%curr_iter)
                torch.save(optimizer.state_dict(),ckpt_path+'optim_%d.pth'%curr_iter)
                return

if __name__ == '__main__':
    main()
