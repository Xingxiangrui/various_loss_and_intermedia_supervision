import os
import shutil
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import util
from torch import nn


# from train import Config
#
# args = Config

class Trainer(object):
    def __init__(self, args, train_dataloader, val_dataloader, optimizer, model, criterion, lr_scheduler):
        self.use_gpu = torch.cuda.is_available()
        self.image_size = args.IMAGE_SIZE
        self.batch_size = args.BATCH_SIZE
        self.evaluate = args.EVALUATE
        self.start_epoch = args.START_EPOCH
        self.max_epochs = args.MAX_EPOCH
        self.workers = args.WORKERS
        self.epoch_step = args.EPOCH_STEP  # fixme: list
        self.device_ids = args.DEVICE_IDS
        self.lr_scheduler = args.LR_SCHEDULER
        self.lr_scheduler_params = args.LR_SCHEDULER_PARAMS
        self.resume_file = args.RESUME
        self.lr = args.LR
        self.lrp = args.LRP
        self.loss_type = args.LOSS_TYPE
        if self.loss_type == 'DeepMarLoss':
            self.deepmar_loss = args.DEEPMAR_LOSS
        self.arch = args.MODEL
        self.save_model_path = args.SAVE_MODEL_PATH
        self.difficult_examples = True

        # display parameters
        self.print_freq = args.PRINT_FREQ
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.best_score = 0.
        self.filename_previous_best = None

        # meters
        # fixme add different loss
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.output_meter_loss=tnt.meter.AverageValueMeter()
        self.supplement_meter_loss=tnt.meter.AverageValueMeter()
        self.batch_time = tnt.meter.AverageValueMeter()
        self.data_time = tnt.meter.AverageValueMeter()
        self.ap_meter = util.AveragePrecisionMeter(self.difficult_examples)

    def resume(self, model):
        print("=> loading checkpoint '{}'".format(self.resume_file))
        checkpoint = torch.load(self.resume_file)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        # model.load_state_dict(checkpoint['state_dict'])
        # fixme!!!!!!
        model.load_state_dict({'module.' + k: v for k, v in checkpoint['state_dict'].items()})
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.resume_file, checkpoint['epoch']))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                decay = 0.1 ** (epoch // self.epoch_step)
                param_group['lr'] = decay * self.lrp  # fixme

                print('backbone learning rate', param_group['lr'])
            if i == 1:
                decay = 0.1 ** (epoch // self.epoch_step)
                param_group['lr'] = decay * self.lr  # fixme
                print('head learning rate', param_group['lr'])

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()
        self.meter_loss.reset()
        self.supplement_meter_loss.reset()
        self.output_meter_loss.reset()
        self.batch_time.reset()
        self.data_time.reset()
        self.ap_meter.reset()

        begin = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            data_time_batch = time.time() - begin
            self.data_time.add(data_time_batch)

            # on start batch
            target_gt = target.clone()
            target[target == 0] = 1  # 0 means difficult? TODO
            target[target == -1] = 0
            image = input[0]  # image,filename, onehot/embedding
            filename = input[1]
            embedding = input[2]

            # on forward
            image.require_grad = True
            target.require_grad = True
            embedding.require_grad = True
            if torch.cuda.is_available():
                image = image.float().cuda(async=True)
                target = target.float().cuda(async=True)
                embedding = embedding.float().cuda(async=True)

            # fixme  add supplement loss compute output
            if(self.arch=='group_clsgat_with_supple_loss'):
                output,supplement_out = model(image, embedding)
            else:
                output = model(image, embedding)

            # fixme add supplement loss to compute new loss
            if (self.arch == 'group_clsgat_with_supple_loss'):
                if self.loss_type == 'DeepMarLoss':
                    weights = self.deepmar_loss.weighted_label(target.detach())
                    if torch.cuda.is_available():
                        weights = weights.cuda()
                    output_loss = criterion(output, target, weight=weights)
                    supplement_loss=criterion(supplement_out, target, weight=weights)
                    loss=output_loss+0.1*supplement_loss
                else:
                    output_loss=criterion(output, target)
                    supplement_loss=criterion(supplement_out, target)
                    loss = output_loss+0.1*supplement_loss
            else:
                if self.loss_type == 'DeepMarLoss':
                    weights = self.deepmar_loss.weighted_label(target.detach())
                    if torch.cuda.is_available():
                        weights = weights.cuda()
                    loss = criterion(output, target, weight=weights)
                else:
                    loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # measure elapsed time
            batch_time_current = time.time() - begin
            self.batch_time.add(batch_time_current)
            begin = time.time()
            # measure accuracy

            # fixme  add output loss and supplement loss
            loss_batch = loss.item()
            self.meter_loss.add(loss_batch)
            if (self.arch == 'group_clsgat_with_supple_loss'):
                supplement_loss_batch=supplement_loss.item()
                output_loss_batch=output_loss.item()
                self.supplement_meter_loss.add(supplement_loss_batch)
                self.output_meter_loss.add(output_loss_batch)

            # measure mAP
            self.ap_meter.add(output.detach(), target_gt.detach())
            if i % self.print_freq == 0:
                # fixme add output loss and supplement loss
                loss = self.meter_loss.value()[0]
                if (self.arch == 'group_clsgat_with_supple_loss'):
                    supplement_loss=self.supplement_meter_loss.value()[0]
                    output_loss=self.output_meter_loss.value()[0]

                batch_time = self.batch_time.value()[0]
                data_time = self.data_time.value()[0]
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    epoch, i, len(data_loader),
                    batch_time_current=batch_time_current,
                    batch_time=batch_time, data_time_current=batch_time_current,
                    data_time=data_time, loss_current=loss_batch, loss=loss))
        # evaluate for training
        map = 100 * self.ap_meter.value().mean()
        loss = self.meter_loss.value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        # OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        # fixme print output loss and supplement loss
        if (self.arch == 'group_clsgat_with_supple_loss'):
            print('Epoch: [{0}]\t'
                  'Loss {loss:.4f}\t'
                  'output_loss {output_loss:.4f}'
                  'supplement_loss {supplement_loss:.4f}'
                  'mAP {map:.3f}'.format(epoch, loss=loss, output_loss=output_loss,supplement_loss=supplement_loss,map=map))
        else:
            print('Epoch: [{0}]\t'
                  'Loss {loss:.4f}\t'
                  'mAP {map:.3f}'.format(epoch, loss=loss, map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

    def validate(self, dataloader, model, criterion):
        # switch to evaluate mode
        self.model.eval()
        self.meter_loss.reset()
        self.batch_time.reset()
        self.data_time.reset()
        self.ap_meter.reset()

        begin = time.time()
        for i, (input, target) in enumerate(dataloader):
            # measure data loading time
            data_time_batch = time.time() - begin
            self.data_time.add(data_time_batch)
            # on start batch
            target_gt = target.clone()
            target[target == 0] = 1  # 0 means difficult? TODO
            target[target == -1] = 0
            image = input[0]  # image,filename, onehot/embedding
            filename = input[1]
            embedding = input[2]
            if torch.cuda.is_available():
                image = image.float().cuda(async=True)
                target = target.float().cuda(async=True)
                embedding = embedding.float().cuda(async=True)

            # fixme  add supplement loss
            if (self.arch == 'group_clsgat_with_supple_loss'):
                # on forward
                with torch.no_grad():
                    # compute output
                    output,supplement_out = model(image, embedding)

                    if self.loss_type == 'DeepMarLoss':
                        weights = self.deepmar_loss.weighted_label(target)  # todo
                        if torch.cuda.is_available():
                            weights = weights.cuda()
                        output_loss = criterion(output, target, weight=weights)
                        supplement_loss=criterion(supplement_out, target, weight=weights)
                        loss=output_loss+supplement_loss
                    else:
                        output_loss = criterion(output, target)
                        supplement_loss=criterion(supplement_out, target)
                        loss=output_loss+supplement_loss
            else:
                # on forward
                with torch.no_grad():
                    # compute output
                    output = model(image, embedding)

                    if self.loss_type == 'DeepMarLoss':
                        weights = self.deepmar_loss.weighted_label(target)  # todo
                        if torch.cuda.is_available():
                            weights = weights.cuda()
                        loss = criterion(output, target, weight=weights)
                    else:
                        loss = criterion(output, target)

            # measure elapsed time
            batch_time_current = time.time() - begin
            self.batch_time.add(batch_time_current)
            begin = time.time()
            # measure accuracy

            loss_batch = loss.item()
            self.meter_loss.add(loss_batch)
            # fixme add supplemeent loss and output loss
            if (self.arch == 'group_clsgat_with_supple_loss'):
                supplement_loss_batch=supplement_loss.item()
                output_loss_batch=output_loss.item()
                self.supplement_meter_loss.add(supplement_loss_batch)
                self.output_meter_loss.add(output_loss_batch)

            # measure mAP
            self.ap_meter.add(output.detach(), target_gt.detach())
            if i % self.print_freq == 0:
                loss = self.meter_loss.value()[0]
                if (self.arch == 'group_clsgat_with_supple_loss'):
                    supplement_loss=self.supplement_meter_loss.value()[0]
                    output_loss=self.output_meter_loss.value()[0]
                batch_time = self.batch_time.value()[0]
                data_time = self.data_time.value()[0]
                if (self.arch == 'group_clsgat_with_supple_loss'):
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                          'Data {data_time_current:.3f} ({data_time:.3f})\t'
                          'Loss {loss_current:.4f} ({loss:.4f})\t'
                          'supplement_loss {supplement_loss:.4f}\t'
                          'output_loss {output_loss:.4f}'.format(
                        i, len(dataloader), batch_time_current=batch_time_current,
                        batch_time=batch_time, data_time_current=data_time_batch,
                        data_time=data_time, loss_current=loss_batch, loss=loss,
                        supplement_loss=supplement_loss,output_loss=output_loss))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                          'Data {data_time_current:.3f} ({data_time:.3f})\t'
                          'Loss {loss_current:.4f} ({loss:.4f})'.format(
                        i, len(dataloader), batch_time_current=batch_time_current,
                        batch_time=batch_time, data_time_current=data_time_batch,
                        data_time=data_time, loss_current=loss_batch, loss=loss))
        # evaluate for validation
        map = 100 * self.ap_meter.value().mean()
        loss = self.meter_loss.value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)

        print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return map

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self.save_model_path is not None:
            filename_ = filename
            filename = os.path.join(self.save_model_path, filename_)
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self.save_model_path is not None:
                filename_best = os.path.join(self.save_model_path, filename_best)
            shutil.copyfile(filename, filename_best)
            if self.save_model_path is not None:
                if self.filename_previous_best is not None:
                    os.remove(self.filename_previous_best)
                filename_best = os.path.join(self.save_model_path,
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.filename_previous_best = filename_best

    def run(self):
        if os.path.isfile(self.resume_file):
            self.resume(self.model)
        else:
            print("=> no checkpoint found at '{}'".format(self.resume_file))

        for epoch in range(self.start_epoch, self.max_epochs):
            if self.lr_scheduler is None:
                self.adjust_learning_rate(self.optimizer, epoch)

                # train for one epoch
            self.train(self.train_dataloader, self.model, self.criterion, self.optimizer, epoch)
            # evaluate on validation set
            best_score = self.validate(self.val_dataloader, self.model, self.criterion)
            # fixme=====================
            if self.lr_scheduler is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if i == 0:
                        print('scheduler backbone learning rate', param_group['lr'])
                    if i == 1:
                        print('scheduler head learning rate', param_group['lr'])
            # fixme=========================

            # remember best prec@1 and save checkpoint
            is_best = best_score > self.best_score
            self.best_score = max(best_score, self.best_score)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.arch,
                'state_dict': self.model.module.state_dict() if torch.cuda.is_available() else self.model.state_dict(),
                'best_score': self.best_score,
            }, is_best)

            print(' *** best={best:.3f} ***'.format(best=self.best_score))
        return
