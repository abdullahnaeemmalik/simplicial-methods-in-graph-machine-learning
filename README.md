This repository has code for my PhD Dissertation titled _Simplicial Methods in Graph Machine Learning_. In this thesis, we propose different neural network architectures that use Simplicial Sets. Simplicial Sets are combinatorial objects that generalize directed graphs from binary relations to higher relations. These architectures, therefore, may fall under the purview of Graph Neural Networks, but with graphs lifted. In particular, these architectures may be ported into pipelines of off-the-shelf implementations of graph neural networks.

#Theoretical justifications

Past experience tells us that graph neural networks need to be made undirected for their performance to not suffer. Theory tells us something otherwise<a id="fn1" href="#footnote1" class="footnote">1</a>. Now we know that the natural asymmetry in directed graphs, once quantified and made crucial part of the message passing, really does perform better. By extension, higher relations based on directed graphs perform better than their undirected counterparts. Think of these directed higher relations as ``oriented communities''.

<div id="footnotes">
  <p id="footnote1"><a href="#fn1" class="backlink">1</a>: Sometimes, we rely on empirical evidence too much and dismiss theory. Sometimes, we shouldn't. </p>
</div>
