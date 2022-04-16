
import os
import time
import torch

from torch.nn import Module
from torch.utils.data import DataLoader

from lantern.utils.logging import SimpleLogger
from lantern.utils.visualization import SceneFittingSummaryWriter

from tqdm.autonotebook import tqdm

device = torch.device("cuda:0")


class ModelTrainer():
    def __init__(self,
                 model: Module,
                 dataloader: DataLoader,
                 output_prefix='latest',
                 logger_class=SimpleLogger,
                 summary_writer_class=SceneFittingSummaryWriter) -> None:

        self.model = model.cuda()
        self.dataloader = dataloader
        self.output_prefix = output_prefix

        if dataloader is None:
            raise Exception('undefined dataloader')

        if model is None:
            raise Exception('undefined model')

        # Make sure our output directories exist
        experiments_path = '/tmp/lantern/experiments'
        self.output_dir = os.path.join(experiments_path, self.output_prefix)
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.logs_dir = os.path.join(self.output_dir, 'logs')

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self._step = 0
        self._epoch = 0
        self._loss = 0.0
        self._input_dict = {}
        self._output_dict = {}        

        self.logger = logger_class()

        self.summary_writer = summary_writer_class(
            self, logger=self.logger, logs_dir=self.logs_dir)

    def train(self,
              loss_fn=None,
              learning_rate=0.01,
              num_epochs=1,
              steps_until_summary=100):

        if loss_fn is None:
            self.logger.error('undefined loss function')
            return

        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)

        total_steps = 0
        dataloader = self.dataloader

        with tqdm(total=len(dataloader) * num_epochs) as progress_bar:

            train_losses = []

            for epoch in range(num_epochs):

                for step, (input_dict, output_dict) in enumerate(dataloader):

                    start_time = time.time()

                    # Move tensors to the GPU
                    input_dict = {key: value.cuda()
                                  for key, value in input_dict.items()}

                    expected_output_dict = {key: value.cuda()
                                   for key, value in output_dict.items()}

                    # Compute model output
                    predicted_output_dict = self.model(input_dict)

                    losses = loss_fn(predicted_output_dict, expected_output_dict)

                    # Sum loss functions
                    train_loss = 0.
                    for loss_name, loss_val in losses.items():
                        train_loss += loss_val.mean()

                    train_losses.append(train_loss.item())

                    # Reset gradients
                    optimizer.zero_grad()
                    train_loss.backward()

                    optimizer.step()

                    torch.cuda.empty_cache()

                    # Update stuff
                    progress_bar.update(1)
                    total_steps += 1

                    if not total_steps % steps_until_summary:
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.checkpoints_dir,
                                                'model_current.pth')
                                   )
                        tqdm.write("Epoch %d, Total loss %0.6f,"
                                   "iteration time %0.6f" % (
                                       epoch, train_loss,
                                       time.time() - start_time)
                                   )

                        self._step = total_steps
                        self._epoch = epoch
                        self._loss = train_loss
                        self._input_dict = input_dict
                        self._expected_output_dict = expected_output_dict
                        self._predicted_output_dict = predicted_output_dict

                        self.summary_writer.write_summary()

            torch.save(self.model.state_dict(), os.path.join(
                self.models_dir, 'model.pth'))

            self.logger.success('\nModel saved to:',
                                os.path.join(self.models_dir, 'model.pth'))

    def summary(self):

        summary = {}

        summary['step'] = self._step
        summary['epoch'] = self._epoch
        summary['loss'] = self._loss
        summary['input_dict'] = self._input_dict
        summary['expected_output_dict'] = self._expected_output_dict
        summary['predicted_output_dict'] = self._predicted_output_dict

        return summary
