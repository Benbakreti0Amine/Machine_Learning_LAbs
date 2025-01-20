% Facts: Ingredients
ingredient(farine).
ingredient(beurre).
ingredient(oeuf).
ingredient(sel).
ingredient(poires).
ingredient(abricots).
ingredient(agrumes).
ingredient(cerises).

% Rules: Pâte and Tarte Recipes
pate :- ingredient(farine), ingredient(beurre), ingredient(oeuf), ingredient(sel).

tarte_aux_poires :- pate, ingredient(poires).
tarte_aux_abricots :- pate, ingredient(abricots).
tarte_aux_agrumes :- pate, ingredient(agrumes).
tarte_aux_cerises :- pate, ingredient(cerises).

% General predicate to group recipes
recette(tarte_aux_poires) :- tarte_aux_poires.
recette(tarte_aux_abricots) :- tarte_aux_abricots.
recette(tarte_aux_agrumes) :- tarte_aux_agrumes.
recette(tarte_aux_cerises) :- tarte_aux_cerises.