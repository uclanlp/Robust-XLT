import os, argparse, pprint, logging, json, shutil
import numpy as np
from tqdm import tqdm
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="pawsx", choices=["pawsx", "xnli"])
parser.add_argument('--input_dir', type=str, default="data_generalized")
parser.add_argument('--output_dir', type=str, default="data_generalized_augment")
parser.add_argument('--neighbors', type=str, default="counterfitted_neighbors.json")
parser.add_argument('--num', type=int, default=10)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

def perturb(sent, neighbors, n=10):
    words = sent.split(' ')
    
    success = False
    p_sents = []
    
    for i in range(n):
        p_words = []
        for word in words:
            if word not in neighbors:
                p_words.append(word)
            elif len(neighbors[word]) == 0:
                p_words.append(word)
            else:
                success = True
                tmp_list = [word] + neighbors[word]
                idx = np.random.randint(0, len(tmp_list))
                p_words.append(tmp_list[idx])
                
        assert len(p_words) == len(words)
        
        p_sents.append(" ".join(p_words))

    return p_sents, success

if not os.path.exists(os.path.join(args.output_dir, args.task)):
    os.makedirs(os.path.join(args.output_dir, args.task))

with open(os.path.join(args.neighbors)) as fp:
    neighbors = json.load(fp)

with open(os.path.join(args.input_dir, args.task, f"train-en.tsv")) as fp:
    lines = fp.readlines()

n_perturb = 0
n_total = len(lines)

with open(os.path.join(args.output_dir, args.task, f"train-en.tsv"), "w") as fp:
    for line in tqdm(lines, ascii=True):
        s1, s2, label = line.split('\t')
        p_sents1, success1 = perturb(s1, neighbors, args.num)
        p_sents2, success2 = perturb(s2, neighbors, args.num)

        if success1 or success2:
            n_perturb += 1

        for p_sent1, p_sent2 in zip(p_sents1, p_sents2):
            fp.write(f"{p_sent1}\t{p_sent2}\t{label}")

logger.info(f"perturb {n_perturb}/{n_total} = {1.0*n_perturb/n_total:.4f}")

if args.task == "pawsx":
    langs = ["en", "de", "es", "fr", "ja", "ko", "zh"]
if args.task == "xnli":
    langs = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    
shutil.copyfile(os.path.join(args.input_dir, args.task, "dev-en.tsv"), os.path.join(args.output_dir, args.task, "dev-en.tsv"))
for lang in langs:
    shutil.copyfile(os.path.join(args.input_dir, args.task, f"test-{lang}.tsv"), os.path.join(args.output_dir, args.task, f"test-{lang}.tsv"))