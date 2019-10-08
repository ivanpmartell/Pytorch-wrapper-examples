from pytorch_lightning import Trainer
from models.FuzzBuzzLightNet import FuzzBuzzModel

model = FuzzBuzzModel(10, 100, 4)

# most basic trainer, uses good defaults
trainer = Trainer()
trainer.fit(model)
