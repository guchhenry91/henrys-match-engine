"""Champions League engine.

Separate from `leagues/` because the competition is not a league. Its teams come
from fifteen countries, most of them play eight matches a season against opponents
they may never meet again, and a third of the 2026/27 field has almost no history
in it at all -- Como none, Viking two matches, Lens six, Stuttgart eight, against
Real Madrid's 182.

THAT IMBALANCE IS THE WHOLE MODELLING PROBLEM. Fitting a strength per club from
its own European record gives Real Madrid a solid estimate and Viking a number
drawn from two games, and the second looks exactly as confident as the first. The
soccer engine already learned this the expensive way: promoted Schalke, with one
player of top-flight history, published as a 72.8% anytime scorer when nothing
else in four leagues beat 50.8%.

THE DRAW POTS ARE THE ANSWER, and they were free all along. UEFA seeds the four
pots by club coefficient, so the pot a club was drawn into IS a published, external
strength estimate that exists for every one of the 36 -- including the ones with no
history. A club with a long European record is fitted from it; a club without one
is shrunk toward its pot; the weight between them is how much evidence it actually
has. Same empirical-Bayes shape as the promoted-club priors in `leagues/`.
"""
