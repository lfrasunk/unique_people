import pandas as pd
from pathlib import Path
import os
import sys
import itertools
import ast
import numpy as np
from fuzzywuzzy import process, fuzz

# Move the working directory to current directory of the script
current_dir = str(Path(sys.path[0]))
os.chdir(current_dir)

# Testing
# sys.argv.append('C:\\Users\\user\\Desktop\\peakdata\\publications_min.csv')

def split_names(target_df, col_name, suffix=''):
  '''
  Function that takes a [target_df] dataframe and splits full names strings
  in column [col_name] into first name, middle name and last name parts,
  adding 'first', 'middle' and 'last' columns. Suffix to the new columns names can be provided

  Args:
      target_df: Dataframe containing the column with names to split.
      col_name: String with a name of the column containing names to split.
      suffix='': String with suffix to add to each of the new columns names.

  Returns:
      Dataframe mimicking input with 3 new columns - 'first', 'middle' and 'last' - containing
      first name, middle name and last name parts of the full names the [col_name].
  '''
  # Separate on space
  target_df['sep'] = target_df[col_name].str.split(' ')
  # First name - first part of the name
  target_df['first'+suffix] = target_df['sep'].str[0]
  # Last name - remaining parts of the name...
  target_df['last'+suffix] = target_df['sep'].str[1:]
  # ... joined to a single string with space separator ...
  target_df['last'+suffix] = target_df['last'+suffix].str.join(' ')
  # ... assume all single letters are middle names
  target_df['middle'+suffix] = target_df['last'+suffix].str.findall(r'[A-Z] ').str.join(' ').str.strip()
  # and get rid of them from the last name
  # will introduce middle names to last name in instances where these are not abbreviated
  target_df['last'+suffix] = target_df['last'+suffix].str.replace(r'[A-Z] ','')
  target_df = target_df.drop(columns='sep')
  # Return extended target_df
  return target_df

def dedupe(df_in, to_fuzz, sort_by, fuzz_th, ret='fuzz', first_letter_match=False, sort_ascending=True):
  '''
  Function returning a dataframe containing deduplicated values of the [ret] column
  with a mapping to original values in a fuzzy-matching approach.
  fuzz.UQRatio is used because it is quite fast and reliable for names (subjective opinion).
  Fuzzy matching is done on all columns provided in [to_fuzz].
  The algorithm is optimized by using sorting and matching only the consequent values,
  instead of comparing all remainign values to each record.
  The algorithm detects groups of matching results (assuming given [fuzz_th] threshold)
  that will be replaced with the most common occurence within the group (next according to sorting).

  Args:
      df_in: Dataframe with columns to find fuzzy-duplicates.
      to_fuzz: List of olumns that will be used to form the string for fuzzy-matching.
      sort_by: List of columns that will be used to sort values (optimization) in the given order.
      sort_ascending=True: List of bool with information if each [sort_by] column
        sholud be sorted ascending, or descending.
        Single Bool value will apply to all [sort_by] columns.
      fuzz_th: Int threshold for the fuzzy matching algorithm (fuzz.UQRatio) score
        to consider two strings matching.
      ret=fuzz': String with a name of the column that will get a mapping to be merged into [df_in].
        'fuzz' will consider the complete strings used for fuzzy matching.
      first_letter_match=False: List of bool for each column in [to_fuzz'].
        Each column with True will get additional check if first letter in that column
        matches between rows. If the firts letters are not equal, fuzzy score is reset to 0,
        breaking a matching group.

  Returns:
      Dataframe with [to_fuzz] columns, 'fuzz' column and '[ret]_fix' column. The last column
      contains deduped values that can be mapped using the [ret] column to the [df_in] dataframe.
  '''
  # limit the loaded dataframe to the columns of interest
  df_fuzz = df_in.loc[:, to_fuzz]
  
  # Args adjustment
  if type(to_fuzz) == str:
    to_fuzz = [to_fuzz]
  if type(sort_by) == str:
    sort_by = [to_fuzz]
  if type(first_letter_match)==bool:
    first_letter_match = [first_letter_match]*len(to_fuzz)
  if type(sort_ascending)==bool:
    sort_ascending = [sort_ascending]*len(sort_by)
    
  # Create the 'fuzz' column - add contents of each [to_fuzz] column, separated with spaces
  df_fuzz['fuzz'] = df_fuzz[to_fuzz[0]]
  if len(to_fuzz)>1:
    for i in range(1,len(to_fuzz)):
      df_fuzz['fuzz'] += ' '+df_fuzz[to_fuzz[i]]
  # Get rid of surplus spaces
  df_fuzz['fuzz'] = df_fuzz['fuzz'].str.replace('\s{2,}',' ')
  
  # Count instances of each unique 'fuzz' value
  df_fuzz['fuzz_count'] = df_fuzz.groupby(['fuzz'])['fuzz'].transform('count')
  # Drop duplicates
  df_fuzz = df_fuzz.drop_duplicates()
  # Init the 'fix' column
  df_fuzz[ret+'_fix'] = df_fuzz[ret]
  
  # If fuzz_th==100, then no fuzzy matching should be performed
  # return df_fuzz at its current state (adjusted to match expected format)
  if fuzz_th==100:
    df_fuzz = df_fuzz[to_fuzz+[ret+'_fix']].drop_duplicates()
    if 'fuzz_fix' in df_fuzz.columns:
      df_fuzz = df_fuzz.rename(columns={'fuzz_fix': 'fuzz'})
    return df_fuzz

  # Typically, first letters in last and first names are correct (typpos usually appear int he middle),
  # sortinng allows optimiization of fuzzy-matching - only pairs of consequent results will be checked
  df_fuzz = df_fuzz.sort_values(by=sort_by)
  # Add new column with [fuzz] from previous row
  df_fuzz['fuzz_prev'] = df_fuzz['fuzz'].shift().fillna('')
  # Fuzzy score
  df_fuzz['fuzz_scr'] = df_fuzz.apply(lambda r: fuzz.UQRatio(r['fuzz'],r['fuzz_prev']), axis=1)
  # Reset score to 0 if first letters doesn't match
  # for consequent rows indicated in the [first_letter_match]
  for i in range(len(to_fuzz)):
    if first_letter_match[i]:
      df_fuzz.loc[
        df_fuzz[to_fuzz[i]].str[0]
        !=
        df_fuzz[to_fuzz[i]].shift().fillna('').str[0]
        , 'fuzz_scr'] = 0
        
  # Create groups of matching results (above [fuzz_th] threshold)
  # Find lower bound
  df_fuzz['group'] = (
    (
      # Between consequent rows score changes from below threshold to above threshold
      # current row <= fuzz_th
      (df_fuzz['fuzz_scr']<=fuzz_th)
      &
      # next row > fuzz_th
      (df_fuzz['fuzz_scr'].shift().fillna(0)>fuzz_th)
    )*1
  )
  # Find upper bound
  df_fuzz['group'] += (
    (
      # Between consequent rows score changes from below threshold to above threshold
      # current row <= fuzz_th
      (df_fuzz['fuzz_scr']<=fuzz_th)
      &
      # previous > fuzz_th
      (df_fuzz['fuzz_scr'].shift(-1).fillna(0)>fuzz_th)
    )*1
  )
  # Run cumsum on the column - will result in increasing number being groupID
  # Furthermore, odd groups contain matches and even groups contain uniques
  #TODO: check if can be other way around
  df_fuzz['group'] = df_fuzz['group'].cumsum()

  '''
  # If fuzz_scr for the next record is above threshold (fuzzy match to the next record),
  # replace the existing score with value from the next row
  df_fuzz.loc[df_fuzz['fuzz_scr'].shift(-1).fillna(0)>fuzz_th, 'fuzz_scr'] = (
    df_fuzz['fuzz_scr'].shift(-1).loc[df_fuzz['fuzz_scr'].shift(-1).fillna(0)>fuzz_th]
  )
  '''
  
  # Sort once again, this time giving priority to 'group' and 'fuzz_count' columns
  df_fuzz = df_fuzz.sort_values(
    by=['group', 'fuzz_count']+sort_by,
    ascending=[True, False]+sort_ascending
  )
  # Drop duplicates to get most common value per group
  most_common_per_group = df_fuzz[['group', ret]].drop_duplicates('group')
  # Merge most common values per group to original dataframe
  df_fuzz = pd.merge(left=df_fuzz, right=most_common_per_group, on='group', suffixes=['','_m'])
  # For groups with matches, put the most common values in the 'fix' column
  df_fuzz.loc[df_fuzz['group']%2==1, ret+'_fix'] = (
    df_fuzz.loc[df_fuzz['group']%2==1, ret+'_m']
  )
  # Limit and adjust the result
  df_fuzz = df_fuzz[to_fuzz+[ret+'_fix']].drop_duplicates()
  if 'fuzz_fix' in df_fuzz.columns:
    df_fuzz = df_fuzz.rename(columns={'fuzz_fix': 'fuzz'})
  return df_fuzz

# Allow kwargs for running the script from command line
kwargs_defaults = {'last': 100, 'first_abb': 92, 'full': 90}
fuzz_ths = {
  kwarg: int(arg.split('=')[-1].strip())
  for kwarg, arg in list(itertools.product(kwargs_defaults, sys.argv))
  if kwarg+'=' in arg.replace(' ','')
}
for kwarg in kwargs_defaults.keys():
  if kwarg not in fuzz_ths.keys():
    fuzz_ths[kwarg] = kwargs_defaults[kwarg]
    
if len(sys.argv)>1:
  # For every file
  for file in sys.argv[1:]:
    print('Current file:', file)
    # Skip kwargs (if provided)
    if any([x+'=' in file.replace(' ','') for x in kwargs_defaults.keys()]):
      continue
    # Load file, only the 'affiliations' and 'authors' columns
    pubs = pd.read_csv(file, usecols=['affiliations', 'authors'])
    
    # --- CLEANING
    print('Cleaning data...')
    # Remove records with missing authors
    pubs = pubs.dropna(subset=['authors'])
    # Convert string representation of a list to list
    pubs['authors_sep'] = pubs['authors'].apply(ast.literal_eval)
    # Number of authors
    pubs['authors_num'] = pubs['authors_sep'].str.len()
    
    # Split the data in two parts:
    # (1) Records with affilations provided
    pubs_affil = pubs.loc[~pubs['affiliations'].isna(), :].copy()
    # Split on ".,", "\" is to escape regex
    pubs_affil['affils'] = pubs_affil['affiliations'].str.split('\.,')
    pubs_affil['affils'] = pubs_affil['affils'].apply(
                          lambda affils: [a for a in affils if ' , ' not in a])
    pubs_affil['affils_len'] = pubs_affil['affils'].str.len()
    # More affils than authors - remove surplus affils
    pubs_affil.loc[pubs_affil['affils_len']>pubs_affil['authors_num'], 'affils'] = (
      pubs_affil.loc[pubs_affil['affils_len']>pubs_affil['authors_num'], :].apply(
        lambda r: r['affils'][:r['authors_num']], axis=1
      )
    )
    # Less affils than authors - add empty affils
    pubs_affil.loc[pubs_affil['affils_len']<pubs_affil['authors_num'], 'affils'] = (
      pubs_affil.loc[pubs_affil['affils_len']<pubs_affil['authors_num'], :].apply(
        lambda r: r['affils']+['']*(r['authors_num']-r['affils_len']), axis=1
      )
    )
    # (2) Records with nan in affilations 
    pubs_no_affil = pubs.loc[pubs['affiliations'].isna(), :].copy()
    # Fill with list of empty strings with the same length as [authors]
    pubs_no_affil['affils'] = pubs_no_affil['authors_num'].apply(lambda al: ['']*al)

    # Convert authors columns from (1) and (2) to lists and add them
    all_authors = np.concatenate(pubs_affil['authors_sep'].values).tolist()
    all_authors += np.concatenate(pubs_no_affil['authors_sep'].values).tolist()
    # Do the same for affils
    all_affils = np.concatenate(pubs_affil['affils'].values).tolist()
    all_affils += np.concatenate(pubs_no_affil['affils'].values).tolist()
    
    # Convert the lists back into a DataFrame
    au_aff = pd.DataFrame(
      list(zip(all_authors, all_affils)),
      columns = ['author', 'affiliation']
    )
    # Clean authors and affils by removing dots and multiple spaces
    au_aff['author'] = (
      au_aff['author']
        .str.replace('\.',' ')
        .str.replace('\s{2,}',' ')
        .str.title()
        .str.strip()
    )
    au_aff['affiliation'] = (
      au_aff['affiliation']
        .str.replace('\.',' ')
        .str.replace('\s{2,}',' ')
        .str.title()
        .str.strip()
    )
    # Drop duplicates
    au_aff = au_aff.drop_duplicates()
    # Split author names into first, middle and last parts
    au_aff = split_names(au_aff, 'author')
    # Add a column with first name abbreviated
    au_aff['fist_abb'] = au_aff['first'].str[0]
    
    # --- FUZZY MATCHING
    print('Fuzzy matching 1/3...')
    # Fuzzy find last name duplicates basing only on the [last] column
    fuzz_last = dedupe(au_aff, to_fuzz=['last'], sort_by=['last'],
                       ret='last', fuzz_th = fuzz_ths['last'])
    # Merge last names mapping to au_aff and apply
    au_aff = pd.merge(left=au_aff, right=fuzz_last, how='left', on='last')
    au_aff['last'] = au_aff['last_fix']
    au_aff = au_aff.drop(columns=['last_fix'])

    print('Fuzzy matching 2/3...')
    # Fuzzy find last name duplicates basing on [fist_abb] [middle] [last] columns
    fuzz_fist_abb = dedupe(au_aff, to_fuzz=['fist_abb', 'middle', 'last'],
                           first_letter_match=[True, False, False],
                           sort_by=['fist_abb', 'last'],
                           ret='last', fuzz_th = fuzz_ths['first_abb'])
    # Merge last names mapping to au_aff and apply
    au_aff = pd.merge(
      left=au_aff, right=fuzz_fist_abb, how='left', on=['fist_abb', 'middle', 'last']
    )
    au_aff['last'] = au_aff['last_fix']
    au_aff = au_aff.drop(columns=['last_fix'])

    print('Fuzzy matching 3/3...')
    # Fuzzy find full name duplicates basing on [first] [middle] [last] columns
    fuzz_full_names = dedupe(au_aff[au_aff['first'].str.len()>1],
                             to_fuzz=['first', 'middle', 'last'],
                             first_letter_match=[True, False, False],
                             sort_by=['first', 'last'],
                             fuzz_th = fuzz_ths['full'])
    fuzz_full_names['fist_abb'] = fuzz_full_names['first'].str[0]

    print('Last touches...')
    # Perform 3 merges for configurations dropping in merge quality
    # Records matched in each step are saved, while unmatched carry over to the next step
    # The purpose is to convert first name abbreviations into full first names
    # (1) Merge on [first] [middle] [last] to [first] [middle] [last]
    au_aff_fml = pd.merge(left=au_aff, right=fuzz_full_names, how='left',
                          on=['first', 'middle', 'last'], suffixes=['','_y'])
    au_aff_fml = au_aff_fml.drop(columns=[col for col in au_aff_fml.columns if '_y' in col])
    # (2) Merge on [first] [middle] [last] to [fist_abb] [middle] [last]
    au_aff_faml = au_aff_fml[au_aff_fml['fuzz'].isna()]
    au_aff_faml = au_aff_faml.drop(columns='fuzz')
    au_aff_faml = pd.merge(left=au_aff_faml, right=fuzz_full_names, how='left',
                           left_on=['first', 'middle', 'last'],
                           right_on=['fist_abb', 'middle', 'last'], suffixes=['','_y'])
    au_aff_faml = au_aff_faml.drop(columns=[col for col in au_aff_faml.columns if '_y' in col])
    # (3) Merge on [first] [last] to [fist_abb] [last]
    au_aff_fl = au_aff_faml[au_aff_faml['fuzz'].isna()]
    au_aff_fl = au_aff_fl.drop(columns='fuzz')
    au_aff_fl = pd.merge(left=au_aff_fl, right=fuzz_full_names, how='left',
                         left_on=['first', 'last'],
                         right_on=['fist_abb', 'last'], suffixes=['','_y'])
    au_aff_fl = au_aff_fl.drop(columns=[col for col in au_aff_fl.columns if '_y' in col])

    # Any remaining records that could not be matched are unique records with no full first name
    # (only first name abbreviations)
    au_aff_u = au_aff_fl[au_aff_fl['fuzz'].isna()]
    au_aff_u = au_aff_u.drop(columns='fuzz')

    # Generate output by joining fuzz_full_names
    # and records that could not be matched to fuzz_full_names (au_aff_u)
    output = (pd.concat([fuzz_full_names['fuzz'], au_aff_u['author']], ignore_index=True)
      .drop_duplicates()
      .to_frame()
      .rename(columns={0: 'author'})
    )
    # Split author names
    output = split_names(output, 'author').drop(columns=['author', 'middle'])
    output = output.rename(columns={'first': 'firstname', 'last': 'lastname'})
    # Save to file
    output.to_csv('unique_people.csv', index=False)