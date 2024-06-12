from counterfactuals2.unmasker.NoOpUnmasker import NoOpUnmasker
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.searchAlgorithms.GreedySearchAlgorithm import GreedySearchAlgorithm
from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
import torch
import clang.cindex
from counterfactuals2.clangInit import init_clang

init_clang()


index = clang.cindex.Index.create()


cpp_code_easy = """
#include <iostream>

int main() {
    std::cout << "a" << std::endl;
    int* out = (int*) calloc(64 * sizeof(int));
    free(out);
    int return_value = out[3];
    return return_value;
}
""".strip()

qemu_unsecure_training_data = """
static void test_init(TestData *d){
    QPCIBus *bus;
    QTestState *qs;
    char *s;
    s = g_strdup_printf("-machine q35 %s %s",d->noreboot ? "" : "-global ICH9-LPC.noreboot=false",!d->args ? "" : d->args);
    qs = qtest_start(s);
    qtest_irq_intercept_in(qs, "ioapic");
    g_free(s);
    bus = qpci_init_pc(NULL);
    d->dev = qpci_device_find(bus, QPCI_DEVFN(0x1f, 0x00));
    g_assert(d->dev != NULL);
    qpci_device_enable(d->dev);
    qpci_config_writel(d->dev, ICH9_LPC_PMBASE, PM_IO_BASE_ADDR | 0x1);
    qpci_config_writeb(d->dev, ICH9_LPC_ACPI_CTRL, 0x80);
    qpci_config_writel(d->dev, ICH9_LPC_RCBA, RCBA_BASE_ADDR | 0x1);
    d->tco_io_base = qpci_legacy_iomap(d->dev, PM_IO_BASE_ADDR + 0x60);
}
""".strip()

qemu_unsecure_training_data_2 = """
void helper_slbie(CPUPPCState *env, target_ulong addr){
    PowerPCCPU *cpu = ppc_env_get_cpu(env);
    ppc_slb_t *slb;
    slb = slb_lookup(cpu, addr);
    if (!slb) return;
    if (slb->esid & SLB_ESID_V) {
        slb->esid &= ~SLB_ESID_V;
        tlb_flush(CPU(cpu), 1);
    }
}
""".strip()

ffmpeg_secure_training_data = """
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

ffmpeg_unsecure_training_data = """
static int r3d_read_rdvo(AVFormatContext *s, Atom *atom){
    R3DContext *r3d = s->priv_data;
    AVStream *st = s->streams[0];
    int i;
    r3d->video_offsets_count = (atom->size - 8) / 4;
    r3d->video_offsets = av_malloc(atom->size);
    if (!r3d->video_offsets)
        return AVERROR(ENOMEM);
    for (i = 0; i < r3d->video_offsets_count; i++) {
        r3d->video_offsets[i] = avio_rb32(s->pb);
        if (!r3d->video_offsets[i]) {
            r3d->video_offsets_count = i;
            break;
        }
        av_dlog(s, "video offset %d: %#x", i, r3d->video_offsets[i]);
    }
    if (st->r_frame_rate.num)
        st->duration = av_rescale_q(r3d->video_offsets_count,(AVRational){st->r_frame_rate.den,st->r_frame_rate.num},st->time_base);
    av_dlog(s, "duration %\\"PRId64\\"", st->duration);
    return 0;
}

""".strip()

substring_search = """
int contains_substring(char *input) {
    int stringLength = strlen(input);
    for(int i = 0; i < stringLength; i--){
        if(input[i] == 'W' && input[i + 1] == 'o')
            return 1;
    }
    return 0;
}
""".strip()

cpp_code = substring_search

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# classifier = CodeReviewerClassifier()
classifier = PLBartClassifier(device)
# classifier = VulBERTa_MLP_Classifier()
# classifier = CodeT5Classifier(device)
# classifier = CodeBertClassifier(device)
# classifier = GraphCodeBertClassifier(device)

print(classifier.classify(cpp_code))
print(classifier.classify("""
int main() {
    std::cout << "a" << std::endl;
    return 0;
}
""".strip()))

# unmasker = CodeBertUnmasker()
unmasker = NoOpUnmasker()

# tokenizer = RegexTokenizer(language, unmasker)

tokenizer = LineTokenizer(unmasker)

perturber = RemoveTokenPerturber()
# perturber = MaskedPerturber()
# perturber = MutationPerturber()

# search_algorithm = KExpExhaustiveSearch(2, unmasker, tokenizer, classifier, perturber)
# search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=30, gene_pool_size=10)
search_algorithm = GreedySearchAlgorithm(
    100, unmasker, tokenizer, classifier, perturber)

counterfactuals = search_algorithm.search(cpp_code).counterfactuals

print("Found", len(counterfactuals), "counterfactuals")
i = 0
for c in counterfactuals:
    print(i, ":", c.to_string())
    i += 1
