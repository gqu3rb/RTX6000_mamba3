import yaml
from lightning import Trainer
from argparse import Namespace
from argparse import ArgumentParser

# This dataset and model are just for demonstration purposes. You can replace them with your own dataset and model.
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger 

from transformers import AutoTokenizer

from mamba import MambaLLM
from mamba.tiny import TinyStoriesLightning

def main(args : Namespace):
    '''
    Main function for the training script.
    '''
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    tokenizer.add_special_tokens(config['tokenizer']['special_tokens'])
    
    # ****** Load the Mamba model ******
    model = MambaLLM.from_config(args.config, key='llm')
    model.tokenizer = tokenizer
    
    # ***** Load the dataset *****
    dataset = TinyStoriesLightning.from_config(
        args.config,
        tokenizer,
        key='dataset'
    )
    
    # Instantiate a logger and a callback for model checkpointing
    callbk = ModelCheckpoint(**config['ckpt'])
    logger = TensorBoardLogger(**config['logger'])
    
    # Load the trainer
    trainer_conf = {
        **config['train'],
        **vars(args),
        'logger' : logger,
        'callbacks' : callbk,
    }
    trainer_conf.pop('config')
    trainer_conf.pop('batch_size', None) 
    trainer = Trainer(**trainer_conf)
    
    # ****** Train the model ******
    trainer.fit(model, dataset, ckpt_path=config['misc']['resume'])

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Mamba LLM Training Script',
        description='Train a Mamba LLM model on the Tiny Stories dataset.',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the training configuration file.',
    )
    
    # parser = Trainer.add_argparse_args(parser) # type: ignore
    
    args = parser.parse_args()
    
    main(args)