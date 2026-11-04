# Representación de texto y metadatos (v1)

Dimensiones ajustando las transformaciones con todo train (3,842 videos); en validación cruzada se ajustan por fold.

| Bloque | Columnas generadas |
|---|---:|
| TF-IDF title | 4,985 |
| TF-IDF description | 27,344 |
| TF-IDF tags | 5,927 |
| Metadatos (escalados + one-hot) | 32 |
| **Combinado** | **38,288** |

Términos más asociados a cada clase en train (chi², uno contra el resto):

| Clase | tags |
|---|---|
| disturbing | spiderman, superhero, mlg, spidey, spider-man, superheroes, joker, elsa |
| irrelevant | srbija, pop, 2016, sc heerenveen, rap, 1984, comedian, sports |
| restricted | sexy, sex, prankzone tv, playstation 4, sony computer entertainment, hot, #ps4share, hypnosis |
| suitable | tom and jerry, jerry, tom and jerry cartoon, cartoon, kids, animation, animated cartoon (tv genre), tom and jerry full episodes |

| Clase | title |
|---|---|
| disturbing | mlg, mlg peppa, peppa, spiderman, peppa pig, pig, elsa, frozen |
| irrelevant | 2015, 2016, video, official video, audio, 2014, pilot, 2018 |
| restricted | sex, scenes, gta, sex scenes, hottest, 10 hottest, smoking, cunnilingus |
| suitable | jerry, tom, tom and, and jerry, kids, for kids, jerry episode, episode |

