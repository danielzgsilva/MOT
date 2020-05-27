from .options import TrainingOptions
from .trainer import UnSupervisedTrainer

options = TrainingOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = UnSupervisedTrainer(opts)

    trainer.train()
