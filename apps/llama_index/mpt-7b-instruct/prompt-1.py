import time
import openai
openai.api_key = "EMPTY" 
openai.api_base = "http://localhost:8000/v1"

model = "mosaicml/mpt-7b-instruct"

text = "Y Combinator was not the original name. At first we were called Cambridge Seed. But we didn't want a regional name, in case someone copied us in Silicon Valley, so we renamed ourselves after one of the coolest tricks in the lambda calculus, the Y combinator."

prompt_template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"  
            "Some text is provided below. Given the text, extract up to 5 knowledge triplets in the form of " 
            "(subject, predicate, object). Avoid duplicates. \n\n"  
            "### Input: \n"
            "Text: Alice is Bob's mother. \n" 
            "Triplets: \n"
            "    (Alice, is mother of, Bob) \n"
            "Text: Philz is a coffee shop founded in Berkeley in 1982. \n"
            "Triplets: \n"
            "    (Philz, is, coffee shop) \n"
            "    (Philz, founded in, Berkeley) \n"
            "    (Philz, founded in, 1982) \n"
            "Text: This small and colorful book is for children. It was named after the moon, and was gifted to Jack. \n"
            "Triplets: \n"
            "    (book, is for, children)\n"
            "    (book, is, small and colorful) \n"
            "    (small book, is for, children) \n"
#            "    (this small book, is for, children) \n"
            "    (book, named after, moon) \n"
            "    (book, gifted to, Jack) \n"    
            "Text: Nick saw a few dwellings, brightly painted cottages, shining in the sun. They were not ready for guests. \n"
            "Triplets: \n"
            "    (dwellings, are, brightly painted cottages) \n"
#            "    (brightly painted cottages, shine in, the sun) \n"
            "    (dwellings, shine in, sun) \n"
            "    (dwellings, not ready for, guests) \n"
            "    (Nick, saw, dwellings) \n"
#            "    (Nick, saw, brightly painted cottages) \n"
#            "    (Nick, saw, painted cottages) \n"    
#            "--------------------- \n"
            "### Text: {text} \n"
            "### Response:"
)

prompt = prompt_template.format(text=text)

print(prompt)