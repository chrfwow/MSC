from captum.attr import LayerConductance, LayerIntegratedGradients, IntegratedGradients
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset
import json
from captum.attr import visualization, TokenReferenceBase
from transformers import PLBartTokenizer, PLBartForSequenceClassification, AutoTokenizer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "uclanlp/plbart-c-cpp-defect-detection"

tokenizer = AutoTokenizer.from_pretrained(path)

model = PLBartForSequenceClassification.from_pretrained(
    path, output_attentions=True)
model.to(device)
model.eval()
model.zero_grad()


vis = True
show_progress = False


def add_attributions_to_visualizer(attributions, code, pred, pred_ind, label, vis_data_records, delta=0):
    attributions = attributions.squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    assert len(attributions) == len(code)

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,  # word_attributions
        pred,  # pred_prob
        pred_ind,  # pred_class
        label,  # true_class
        '1',  # attr_class
        attributions.sum(),  # attr_score
        code,  # raw_input_ids
        delta))  # convergence_score


class CodeInput(object):
    """A single code example."""

    def __init__(self, input_ids, reference_ids, label):
        self.input_ids = input_ids
        self.reference_ids = reference_ids
        self.label = label


class CodeDataset(Dataset):
    """CodeInputs dataset."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        return torch.tensor(
            self.examples[index].input_ids
        ), torch.tensor(
            self.examples[index].reference_ids
        ), torch.tensor(
            self.examples[index].label
        )


ref_token_id = tokenizer.pad_token_id


def construct_code_input(line, tokenizer: PLBartTokenizer, block_size, label: int):
    # construct input token ids
    tokens = tokenizer.tokenize(line)[:block_size - 2]
    tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
    input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

    # construct reference token ids
    ref_input_ids = [x if (x == tokenizer.bos_token_id or x == tokenizer.eos_token_id) else tokenizer.pad_token_id for x
                     in input_ids]

    # padding
    diff = block_size - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * diff
    ref_input_ids += [tokenizer.pad_token_id] * diff

    return CodeInput(input_ids, ref_input_ids, label)


def construct_attention_mask(input):
    return torch.where(input == tokenizer.pad_token_id, 0, 1)


def predict(inputs):
    attention_mask = construct_attention_mask(inputs)
    return model(input_ids=inputs, attention_mask=attention_mask).logits


layers = []
for l in model.model.decoder._modules["layers"]:
    layers.append(l)

lig = LayerIntegratedGradients(predict, model.model.shared)


# lig = LayerIntegratedGradients(predict, model.encoder)


def ig_attribute(input_indices, baseline, target):
    attributions, delta = lig.attribute(input_indices, baselines=baseline, target=target, n_steps=50,
                                        return_convergence_delta=True)
    attributions = attributions.sum(dim=-1)
    return attributions / torch.norm(attributions), delta


i = 0
n = 1
mismatch = 0

text = """
static int mov_write_minf_tag(AVIOContext *pb, MOVMuxContext *mov, MOVTrack *track){
    int64_t pos = avio_tell(pb);
    int ret;
    avio_wb32(pb, 0);
    ffio_wfourcc(pb, "minf");
    if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO)
        mov_write_vmhd_tag(pb);
    else if (track->enc->codec_type == AVMEDIA_TYPE_AUDIO)
        mov_write_smhd_tag(pb);
    else if (track->enc->codec_type == AVMEDIA_TYPE_SUBTITLE) {
        if (track->tag == MKTAG('t','e','x','t') || is_clcp_track(track)) {
            mov_write_gmhd_tag(pb, track);
        } else {
            mov_write_nmhd_tag(pb);
        }
    } else if (track->tag == MKTAG('r','t','p',' ')) {
        mov_write_hmhd_tag(pb);
    } else if (track->tag == MKTAG('t','m','c','d')) {
        mov_write_gmhd_tag(pb, track);
    }
    if (track->mode == MODE_MOV)
        mov_write_hdlr_tag(pb, NULL);
    mov_write_dinf_tag(pb);
    if ((ret = mov_write_stbl_tag(pb, mov, track)) < 0)
        return ret;
    return update_size(pb, pos);
}
""".strip()
cpp_code_easy = """
int main() {
    int* out = (int*) malloc(64 * sizeof(int));
    free(out);
    return out[3];
}
""".strip()

tokenized = tokenizer(cpp_code_easy.lower())
tokens = tokenized["input_ids"]

eos_index = 0
while len(tokens) < 10:
    tokens.append(tokenizer.pad_token_id)

tokens.insert(0, tokenizer.bos_token_id)
text = []
i = 0
for token in tokens:
    if token == tokenizer.eos_token_id:
        eos_index = i
    i += 1
    text.append(tokenizer._convert_id_to_token(token).lower())

indexed = [tokenizer._convert_token_to_id(t) for t in text]

print("should be <pad>", tokenizer._convert_id_to_token(tokenizer.pad_token_id))
print("must be true", tokenizer.pad_token ==
      tokenizer._convert_id_to_token(tokenizer.pad_token_id))

PAD_IND = tokenizer.pad_token_id
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
input_indices = torch.tensor(tokens, device=device)
reference_indices = token_reference.generate_reference(
    len(indexed), device=device)
reference_indices[0] = tokenizer.bos_token_id
reference_indices[eos_index] = tokenizer.eos_token_id

true_label = 1
labels = torch.tensor([true_label], device=device)
ig_attrs = []
vis_data_records_ig = []
i += 1

input_indices = input_indices.unsqueeze(0)

print("first predict")
# predict
with torch.no_grad():
    logits = model(input_ids=input_indices).logits

pred = logits.argmax().item()
if pred != labels.unsqueeze(0).item():
    mismatch += 1

print("attribute")
# IG
ig_attributions, delta = ig_attribute(
    input_indices, reference_indices.unsqueeze(0), pred)
ig_attrs.append((tokens, ig_attributions.squeeze(0).tolist()))

print("finished")

add_attributions_to_visualizer(ig_attributions, text, pred, pred, true_label,
                               vis_data_records_ig, delta)

v = visualization.visualize_text(vis_data_records_ig)

data = v.data

new_text = "".join([c for c in data if c.isascii()])

with open("data.html", "w") as file:
    file.write(new_text)
