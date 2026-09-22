# flybrain-flappy

A real fruit-fly brain (the Janelia/Google MaleCNS v1.0 connectome) plays Flappy Bird. The brain's wiring is never changed. Only the interface between the game and the brain is fitted.

The extracted sub-circuit (`data/derived/`) is committed so the tests run on a clone. It is
derived from the Janelia/Google **MaleCNS v1.0** connectome, licensed CC-BY — see `docs/DATA.md`
for the source, licence and citation. The raw 500 MB+ files are not committed; run
`python -m flybrain.fetch` to download them.

Work in progress. The full plan is in [docs/PLAN.md](docs/PLAN.md).
