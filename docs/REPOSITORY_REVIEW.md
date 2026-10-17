# Pregled repozitorijuma i priprema commita

Poslednji commit pre sređivanja: `df1fa922ac6ce39ffd937dbf70a0580ab78db34e`, **18. april 2026**, poruka `llm eval script`. Commit je dodao `eval/run_llm_eval.py`.

Pre početka su već postojale izmene u DSL operatorima, evaluaciji, LLM prevodiocu, detektoru, klasifikatoru atributa, relacijama, rendereru i notebook fajlovima. One nisu nastale tokom ovog sređivanja. Završni rezultati su preuzeti iz julskog PDF-a i potvrđeni prezentacijama; stariji lokalni rezultati nisu korišćeni kao završni.

## Šta je organizovano

| Ranije | Sada |
| :--- | :--- |
| `kaggle_*.py` u korenu | `experiments/kaggle/`, bez ponavljanja prefiksa `kaggle_` |
| `colab_detected_eval.py` | `experiments/colab/detected_eval.py` |
| `nl2dsl/train.py` | `experiments/colab/train_nl2dsl.py` |
| `nl2dsl/finetune_humans.py` | `experiments/kaggle/finetune_humans.py` |
| `nl2dsl/kaggle_finetune_para.py` | `experiments/kaggle/finetune_paraphrases.py` |
| `perception/test_attributes.py` | `eval/attribute_eval.py` |
| Tri vizuelne probe u `tests/test_*.py` | `experiments/local/`; stvarni pytest testovi ostaju u `tests/` |
| Srpska i engleska prezentacija | `docs/presentation/`, obe se čuvaju za commit |
| Word dokumenti i PDF verzije | `.local/thesis/` |
| `overview.md`, nacrti poglavlja i stare beleške | `.local/notes/` |
| Stari sažeci rezultata i grafikoni | `.local/results/historical/`, `.local/figures/historical/` |
| Generisane slike iz testova | `.local/test_outputs/` |
| Privremeni notebook i placeholder fajl | `.local/archive/` |
| ZIP paketi iz korena | `.local/archives/` |

`plots/` je ignorisan lokalni izlaz notebooka. Odabrane ilustracije su kopirane u `docs/assets/source/`. README sada koristi originalne slike 4.1 i 6.1 iz ćiriličnog PDF-a, sa prevedenim oznakama. Pet prethodno napravljenih grafikona, njihov generator, izdvojena istorija treninga i zahtevi generatora sklonjeni su u ignorisani `.local/archive/readme-figures-v1/` i ne ulaze u commit.

Notebook izlazi i metapodaci sesije su očišćeni. Komentari u dodatnim skriptama su znatno skraćeni, a uputstva objedinjena u `experiments/README.md`. Ćelijske granice su zadržane tamo gde pomažu radu u cloud notebooku.

## Fajlovi za tvoju proveru

1. **`README.md`**: predstavljanje projekta, završne tabele, objašnjenja i poređenje sa literaturom.
2. **`results/thesis_reported.json`**: prepisane vrednosti iz PDF tabela. Ovo je označen prepis, a ne izmišljeni Kaggle izlaz.
3. **`docs/assets/` i obe prezentacije**: proveri prikaz grafikona i izbor ilustracija. Originalne prezentacije nisu menjane.
4. **`experiments/kaggle/rl_humans.py` i `humans_val_eval.py`**: proveri koje verzije odgovaraju završnim Kaggle sesijama. Njihov interpreter je sačuvan.
5. **`perception/detector.py`, `perception/attribute_extractor.py`, `scene_graph/relations.py`, `dsl/operators.py`**: proveri ranije nekomitovane eksperimentalne izmene. Razlike u odnosu na tekst rada navedene su u `IMPLEMENTATION_REVIEW.md`.
6. **`nl2dsl/call_paraphrase_api.py` i `llm_pseudolabel.py`**: ključ se sada čita iz okruženja. Pronađeni ugrađeni ključ opozovi i zameni novim. Njegova vrednost nije potrebna u repozitorijumu.
7. **`.gitignore` i `requirements.txt`**: proveri politiku praćenja i biblioteke. Zahtevi nisu originalni zaključani Kaggle/Colab environment.

## Šta komitovati

Kod, očišćene notebook fajlove, konfiguraciju, zahteve, testove, primer, README, dokumentaciju, odabrane slike i njihove izvore, prepis završnih rezultata i **obe PPTX prezentacije**. Uključiti i uklanjanje starih praćenih rezultata i generisanih slika iz Git radnog stabla; lokalne kopije su sačuvane.

Git će premeštene fajlove pre staginga često prikazivati kao `D` na staroj putanji i `??` na novoj. To samo po sebi ne znači da je sadržaj izgubljen. Nakon staginga Git pokušava da prepozna preimenovanja; kod većih čišćenja neki ostaju prikazani kao brisanje i dodavanje.

## Šta ne komitovati

| Materijal | Lokalna putanja / obrazac |
| :--- | :--- |
| Datasetovi, konvertovani JSONL i trening TSV | `datasets/` |
| Modeli i kontrolne tačke | `models/`, `checkpoints/`, `*.pt`, `*.safetensors` |
| Word dokumenti i radne PDF verzije | `.local/thesis/`, `*.docx`, `*.pdf` |
| Svi lokalni nacrti, arhive i backup | `.local/`, `*.zip` |
| Izlazi po pitanjima, keš grafova, pseudo-labele | `results/`, osim dva eksplicitno dozvoljena dokumentaciona fajla |
| Generisane test slike i notebook izlazi | `test_outputs/`, `plots/`, `tests/*_output/` |
| Python okruženja i alati pregleda | `venv/`, `.venv/`, `.review-venv/`, `.tools/` |
| Ključevi, lozinke i lokalna podešavanja | `.env*`, `credentials.json`, `*.key`, `.claude/`, `.codex/` |

`.env.example` sadrži samo ime potrebne promenljive i namenjen je commitu. PDF može kasnije biti objavljen kao posebno izabrano izdanje ako to želiš; trenutno ostaje lokalno, uključujući vraćeni `docs/presentation/DIPLOMSKI_RAD_BOGDAN_PDF_JUL.pdf`. Prezentacije su izričito izuzete iz opšteg ignorisanja PPTX fajlova.

## Predlog commit poruke i provera

Promene su ostavljene za pregled: nije napravljen commit niti push.

```bash
git status --short
git diff --stat
git diff --check
git add --dry-run -A
```

Posle pregleda možeš uključiti promene i proveriti tačan sadržaj staginga:

```bash
git add -A
git diff --cached --stat
git diff --cached --name-only
git commit -m "Organize thesis experiments and document final VQA results"
```

Ne koristi `git add -f` za ignorisane lokalne materijale. `.gitignore` ne uklanja sadržaj koji je već bio u starim commitovima; postojeći istorijat nije prepisivan.

## Provere posle sređivanja

- Lokalni testovi bez modela i datasetova: **251 prošao, 10 izostavljeno** izborom markera `not slow and not clevr and not clevr_full`.
- Sintaksa proverena za Python fajlove i kod iz 6 notebook fajlova, uključujući kompatibilnost sintakse sa Python 3.10.
- Provereni lokalni Markdown linkovi, očišćeni notebook izlazi i obrasci tajnih ključeva u fajlovima predviđenim za commit.
- Primer simboličkog izvršavanja daje odgovor `blue`; prevedene slike iz diplomskog vizuelno su proverene.
- GPU trening i puna evaluacija nisu ponavljani. Završne vrednosti u README-u potiču iz PDF-a, uz proveru ključnih tabela u prezentacijama.

## Sačuvani originali

`.local/review/before-cleanup.zip` i `before-cleanup.patch` čuvaju stanje sa početka pregleda, uključujući prvobitne notebook izlaze. Arhiva sadrži i prvobitnu skriptu sa ugrađenim ključem, pa je samo lokalna i ne treba je deliti. `.local/review/moves.json` sadrži početni spisak premeštanja. Dodatno su kasnije premeštene engleska prezentacija i tri lokalne vizuelne probe.
