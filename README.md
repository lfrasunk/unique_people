# unique_people
## _PeakData Data Engineering Task_
Contains Python script that takes .csv files with publications data
and performs three fuzzy-matching deduplications with (customizable thresholds)
to return a list of unique people (authors).

## Usage
- Python 3 is required with the following packages
    - Pandas
    - numpy
    - fuzzywuzzy (python-Levenshtein not required to finish in finite time)

To run:
```sh
python publications_unique.py *args **kwargs
```
*args:
- any number of filenames from the script's location,
- any number of filenames (full path)

**kwargs:
- last: fuzzy matching threshold for last name matching (0-100, default 100),
- first_abb: fuzzy matching threshold for last name + first letter othe first name matching (0-100, default 92),
- full: fuzzy matching threshold for full name matching (0-100, default 90)

## Documentation
The script expects that the .csv file contains 'author' and 'affiliations' columns and is ignoring anythin else.
The data is loaded and processed accordingly to what can be found in the test dataset provided with the task:
- Publication authors are divided into individual people,
    - individual people have their names divided into first name, middle names and last name parts
- affiliations column in the source lacks documentation and is poorly formatted. It is divided into as many records as possible and then limited up to the numbers of authors. The logic here is one affiliation per author.
(this is not really how affiliations in publications work, but the formatting of the affiliations column
suggests such approach)

The affiliations part is ignored after this part, as it appears to not be required in the output (even though task decription suggests otherwise).

Three fuzzy-matching approaches are used to remove duplicates from the list of authors:
1. Considering only on the last names,
2. Considering the first letter of the first name, middle names and the last name,
3. Considering the full name

Each attempt has a customizable fuzzy score threshold to consider two strings matching.
In order to finish te task in finite time, optimization through specific sorting of the data is done.
First then last names are sorted and only consequent rows are compared. The rows are then grouped and the most common value within each matching group is used to replace all other values within the group (this is further fine tuned by specific mapping in the approach no. 3.).

## Shortcomings
Dividing names into first, middle and last parts could be improved. The logic is:
- up to the first space it is the first name,
- next, every single letter indicates middle name abbreviation (all such letters are saved as middle),
- finally, first non-single-letter word indicate beginning of the last name (ends with the end of the whole name)

This approach fails when middle names are not abbreviated.

Fuzzy matching can result in loss of information, by matching similar, but real last names, or male and female versions of a name (e.g. Alessandro and Alessandra). Further fine tuning or bether models could be utilized.




A reporting of potential failure points and bottlenecks
An accounting of the remaining steps needed before putting deploying your code to a production system
