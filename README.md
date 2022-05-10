'true.csv', 'comformer_w_ast.csv', and 'hybrid.out.new' are the raw result in the paper.
'predict.csv' is the new version result in current DeepCom dataset.

This repo uses [javalang](https://github.com/c2nes/javalang), [transformers](https://github.com/huggingface/transformers), [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers), [pytorch](https://pytorch.org/), and [nlgeval](https://github.com/Maluuba/nlg-eval). If the nlg-eval library installation encounters a dependency conflict, you can modify its project and then run setup.py.

Make sure your dataset is in CSV format and has two columns, code in the first column and comments in the second.

For model inputs, **please ensure that the format is [ codeseq  ⊕ \<code> ⊕ sbt ]**. For Java, we implement it in utils.transformer.

# How To Train-From-Scratch

First you need train the tokenize:

```
python train_tokenizer.py
```

where you also need modify the dataset path, vocab_size and others, then the 'vocab.json' and 'merges.txt' will be saved in 'tokenize' folder.

Next you can train, but also need modify the parameters in the 'train.py', such as your own train.csv, valid.csv and test.csv. If you train from scratch, make sure pretrained_model = None.

```
model = BartModel(pretrained_model=None,args=model_args, model_config='config.json', vocab_file="./tokenize")
```

config.json, like BART, is set to a base or large model, where url is 

https://huggingface.co/facebook/bart-base/blob/main/config.json and 

https://huggingface.co/facebook/bart-large/blob/main/config.json

For other parameters, see [simpletransformers](https://simpletransformers.ai/docs/usage/)

Finally run 

```
python train.py
```

# How To Fine-Tune

If you want to fine-tune the model we pretrained in Java, please make sure:

```
model = BartModel(pretrained_model='NTUYG/ComFormer',args=model_args)
```

Then run 

```
python train.py
```

# How To Use

```PYTHON
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("NTUYG/ComFormer")
tokenizer = BartTokenizer.from_pretrained("NTUYG/ComFormer")

code = '''    
public static void copyFile( File in, File out )  
            throws IOException  
    {  
        FileChannel inChannel = new FileInputStream( in ).getChannel();  
        FileChannel outChannel = new FileOutputStream( out ).getChannel();  
        try
        {  
//          inChannel.transferTo(0, inChannel.size(), outChannel);      // original -- apparently has trouble copying large files on Windows  
 
            // magic number for Windows, 64Mb - 32Kb)  
            int maxCount = (64 * 1024 * 1024) - (32 * 1024);  
            long size = inChannel.size();  
            long position = 0;  
            while ( position < size )  
            {  
               position += inChannel.transferTo( position, maxCount, outChannel );  
            }  
        }  
        finally
        {  
            if ( inChannel != null )  
            {  
               inChannel.close();  
            }  
            if ( outChannel != null )  
            {  
                outChannel.close();  
            }  
        }  
    }
    '''
code_seq, sbt = utils.transformer(code) #can find in https://github.com/NTDXYG/ComFormer
input_text = ' '.join(code_seq.split()[:256]) + ' '.join(sbt.split()[:256])
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=2.0,
    max_length=30,
    min_length=2,
    num_beams=5,
)
comment = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
print(comment)
```

# BibTeX entry and citation info

```
@misc{yang2021comformer,
      title={ComFormer: Code Comment Generation via Transformer and Fusion Method-based Hybrid Code Representation}, 
      author={Guang Yang and Xiang Chen and Jinxin Cao and Shuyuan Xu and Zhanqi Cui and Chi Yu and Ke Liu},
      year={2021},
      eprint={2107.03644},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

