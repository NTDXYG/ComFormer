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
input_text = code_seq + sbt
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, 		truncation=True)
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

