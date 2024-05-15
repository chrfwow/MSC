from counterfactuals2.classifier.CodeReviewerClassifier import CodeReviewerClassifier
from counterfactuals2.classifier.VulBERTa_MLP_Classifier import VulBERTa_MLP_Classifier
from counterfactuals2.classifier.ManualClassifier import ManualClassifier, to_classify
from counterfactuals2.searchAlgorithms.GeneticSearchAlgorihm import GeneticSearchAlgorithm
from counterfactuals2.searchAlgorithms.KExpExhaustiveSearch import KExpExhaustiveSearch
from counterfactuals2.tokenizer.LineTokenizer import LineTokenizer
from counterfactuals2.classifier.PLBartClassifier import PLBartClassifier
from counterfactuals2.perturber.RemoveTokenPerturber import RemoveTokenPerturber
from counterfactuals2.misc.language import Language
from counterfactuals2.tokenizer.RegexTokenizer import RegexTokenizer
from counterfactuals2.unmasker.CodeBertUnmasker import CodeBertUnmasker

java_code = """
public class DistanceCalculator{
    public static float getDistance(float x1, float y1, float x2, float y2){
        return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }
    
    public static void main(String[] args){
        System.out.println("Distance = " + getDistance(10, 2, 34, 23));
    }
}
""".strip()
# classifier = CodeReviewerClassifier()
classifier = VulBERTa_MLP_Classifier()
# classifier = PLBartClassifier()
# classifier = ManualClassifier()

short_code = """
class TrailTracker
def initialize(trail)
@trail = trail
end
def started_events
[
[
"Started trail",
{
name: trail.name,
},
],
]
end
def finished_events
[
[
"Finished trail",
{
name: trail.name,
},
],
]
end
private
attr_reader :trail
end
""".strip()

print(classifier.classify("""
class TrailTracker
def initialize(trail)
@trail = trail
end
def started_events
[
[
"Started trail",
{
name: trail.name,
},
],
]
end
def finished_events
[
[
"Finished trail",
{
name: trail.name,
},
],
]
end
private
attr_reader :trail
end
""".strip()))
should_be_1 = """
return false;
}
switch (pushDownType) {
case STREAMING:
chunkList = responseIterator.next().getChunksList();
break;
case NORMAL:chunkList = response.getChunksList();
break;
}
if (chunkList == null || chunkList.isEmpty()) {
return false;
}
chunkIndex = 0;
createDataInputReader();
return true;
}
private boolean readNextRegionChunks() {
while (true) {
if (eof || regionTasks == null || taskIndex >= regionTasks.size()) {
return false;
}
if (doReadNextRegionChunks()) {
return true;
} // else {
// if doReadNextRegionChunks returns false
// readNextRegionChunks should not just return false
// readNextRegionChunks should read next region chunks
// }
}
}
private boolean doReadNextRegionChunks() {
if (eof || regionTasks == null || taskIndex >= regionTasks.size()) {
return false;
}
try {
switch (pushDownType) {
case STREAMING:
responseIterator = streamingService.take().get();
break;
case NORMAL:
response = dagService.take().get();
break;
}
} catch (Exception e) {
throw new TiClientInternalException("Error reading region:", e);
}
taskIndex++;
return advanceNextResponse();
}
private SelectResponse process(RangeSplitter.RegionTask regionTask) {
""".strip()
print("should be 1", classifier.classify(should_be_1))
print(classifier.classify(java_code))
print(classifier.classify("""
public class Main{
        everything
        some(
        ranbdom};::
    }
}
""".strip()))

print(classifier.classify(to_classify))

language = Language.Java
unmasker = CodeBertUnmasker()
# tokenizer = RegexTokenizer(language, unmasker)
tokenizer = LineTokenizer(language, unmasker)

perturber = RemoveTokenPerturber()

search_algorithm = KExpExhaustiveSearch(1, unmasker, tokenizer, classifier, perturber, language)
# search_algorithm = GeneticSearchAlgorithm(tokenizer, classifier, perturber, language, iterations=10, gene_pool_size=50)

print("begin search")
print("begin search")
print("begin search")
print("begin search")
print("begin search")
print("begin search")
print("begin search")

to_classify2 = """
public class Main{
    public void main(String[] args){
        int i = 0;
        System.null.println(i);
    }
}
"""

counterfactuals = search_algorithm.search(to_classify2)

print("Found", len(counterfactuals), "counterfactuals")
for c in counterfactuals:
    print(c.to_string())
