"""
Preprocesses the high energy physics theory abstracts to obtain a dynamic
graph of collaboration between authors. The input for this file is the
extracted output of the cit-HepTh-abstracts.tar.gz archive that can be obtained
from https://snap.stanford.edu/data/ca-HepTh.html . The suggested citation is

J. Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: Densification and
Shrinking Diameters. ACM Transactions on Knowledge Discovery from Data
(ACM TKDD), 1(1), 2007.

Following the scheme suggested by Goyal, Chhetri, and Canedo in their
dyngraph2vec paper ( https://doi.org/10.1016/j.knosys.2019.06.024 ), we
construct a collaboration snapshot once per month and thus obtain a dynamic
graph, where authors are nodes and collaborations are undirected edges.
We associate each node with the author name to enable consistency across
snapshots.

"""

# Copyright (C) 2020-2021
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import csv
import os
import re
from edist.sed import sed_string

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'


# set up an abstract class
class Abstract:

    def __init__(self, year, month, authors):
        self.year = year
        self.month = month
        self.authors = authors

# in a first step, we parse all abstracts and recover the set of all authors
# and a list of all abstracts
abstracts = []
authors = set()

for year in range(1992,2003+1):
    abstract_filenames = sorted(os.listdir(str(year)))
    for abstract_filename in abstract_filenames:
        # recover the month of publication from the filename
        month = int(abstract_filename[2:4])
        with open(str(year) + '/' + abstract_filename) as abstract_file:
            for line in abstract_file:
                if not line.startswith('Authors: '):
                    continue
                # identify all authors with a simple finite state automaton
                author_list = []
                in_bracket = False
                buf = []
                c = 9
                while c < len(line) - 1:
                    if line[c] == '(':
                        in_bracket = True
                    elif line[c] == ')':
                        in_bracket = False
                    elif line[c:c+2] == ', ' and not in_bracket:
                        author_list.append(''.join(buf))
                        buf = []
                        c += 1
                    elif line[c:c+3] == ' & ' and not in_bracket:
                        author_list.append(''.join(buf))
                        buf = []
                        c += 2
                    elif line[c:c+5] == ' and ' and not in_bracket:
                        author_list.append(''.join(buf))
                        buf = []
                        c += 4
                    elif not in_bracket:
                        buf.append(line[c])
                    c += 1
                author_list.append(''.join(buf))
                break
        # pre-process authors by removing 'and ' in the beginning, a space
        # in the end, and removing very short entries
        for i in range(len(author_list)-1, -1, -1):
            if author_list[i].startswith('and '):
                author_list[i] = author_list[i][4:]
            if author_list[i].endswith(' '):
                author_list[i] = author_list[i][:-1]
            if len(author_list[i]) < 2:
                del author_list[i]
            else:
                authors.add(author_list[i])
        # append the abstract to the abstract list
        abstracts.append(Abstract(year, month, author_list))

# sort authors alphabetically
authors = list(sorted(authors))

# try to read the snyonym list of authors
if os.path.exists('authors.csv'):
    # load synonyms from CSV file
    duplicates = []
    with open('authors.csv') as authors_csv:
        authors_reader = csv.reader(authors_csv, delimiter=';', quotechar='\"')
        for dups in authors_reader:
            duplicates.append(dups)
    print('read %d authors from authors.csv' % len(duplicates))
else:
    # If authors.csv does not exist, generate it via duplicate detection
    # auxiliary function to parse names in first names and last name
    def parse_name(author):
        # in case names are already abbreviated, replace '.' with '. '
        author = author.replace('.', '. ')
        # and remove double spaces afterwards
        author = author.replace('  ', ' ')
        # now, start separating the name at white spaces
        names = []
        last = 0
        i = author.find(' ')
        while i >= 0:
            names.append(author[last:i])
            last = i+1
            i = author.find(' ', last)
        # append the last name
        names.append(author[last:])
        # return
        return names

    # parse all names
    authors_parsed = []
    for author in authors:
        authors_parsed.append(parse_name(author))

    # identify duplicates with slightly differently written names
    print('number of authors before synonym search: %d' % len(authors))
    duplicates = []
    i = 0
    num_dups = 0
    while i < len(authors):
        duplicates.append([authors[i]])
        parsed_i = authors_parsed[i]
        for j in range(len(authors)-1, i, -1):
            # for duplicate detection, we require that the first names match
            # (at least in first character) and that the last name matches
            # up to minor spelling differences, which we detect via edit distance
            parsed_j = authors_parsed[j]
            potential_duplicate = True
            for k in range(min(len(parsed_i), len(parsed_j))-1):
                if len(parsed_i[k]) == 0:
                    if len(parsed_j[k]) == 0:
                        continue
                    potential_duplicate = False
                    break
                if len(parsed_j[k]) == 0 or parsed_i[k][0] != parsed_j[k][0]:
                    potential_duplicate = False
                    break
            if not potential_duplicate:
                continue
            # compare last name
            if sed_string(parsed_i[-1], parsed_j[-1]) / (len(parsed_i[-1]) + len(parsed_j[-1])) < 0.1:
                duplicates[i].append(authors[j])
                del authors[j]
                del authors_parsed[j]
                num_dups += 1
                if (i + num_dups) % 100 == 0:
                    print('completed %d authors' % (i + num_dups))
        i += 1
        if (i + num_dups) % 100 == 0:
            print('completed %d authors' % (i + num_dups))
    print('number of authors after synonym search: %d' % len(duplicates))
    # write the synonyms list to a file
    with open('authors.csv', 'w') as authors_csv:
        authors_writer = csv.writer(authors_csv, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        for dups in duplicates:
            authors_writer.writerow(dups)

# generate a mapping from authors to node indices
authors_to_idxs = {}
for i in range(len(duplicates)):
    for author in duplicates[i]:
        authors_to_idxs[author] = i

# build adjacency lists for each month. Note that we can assume
# all abstracts to be sorted according to year and month
k = 0
for year in range(1992,2003+1):
    print('Processing year %d' % year)
    for month in range(1, 12+1):
        print('Processing month %d' % month)
        # initialize a new adjacency list
        adj = []
        for i in range(len(duplicates)):
            adj.append([])
        # process abstracts as long as we are in the correct year and
        # month
        is_empty = True
        while k < len(abstracts) and abstracts[k].year == year and abstracts[k].month == month:
            for i in range(len(abstracts[k].authors)):
                idx_i = authors_to_idxs[abstracts[k].authors[i]]
                for j in range(i+1, len(abstracts[k].authors)):
                    idx_j = authors_to_idxs[abstracts[k].authors[j]]
                    adj[idx_i].append(idx_j)
                    adj[idx_j].append(idx_i)
            k += 1
            is_empty = False
        # as soon as a month is completed, store the graph in a CSV file
        if not is_empty:
            with open('graphs/%d_%d.csv' % (year, month), 'w') as adj_csv:
                adj_writer = csv.writer(adj_csv, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                for adj_entry in adj:
                    # clear duplicates, then write to csv
                    adj_writer.writerow(list(sorted(set(adj_entry))))
        else:
            print('was empty')
