from typing import Tuple

from graphdatascience import GraphDataScience
from numpy.typing import ArrayLike
from ogb.nodeproppred import NodePropPredDataset
import pandas as pd
import numpy as np
from pandas import DataFrame


def make_set_clause(prop_names: ArrayLike, element_name='n', item_name='rec'):
    clause_list = []
    for prop_name in prop_names:
        clause_list.append(f'{element_name}.{prop_name} = {item_name}.{prop_name}')
    return 'SET ' + ', '.join(clause_list)


def make_node_merge_query(node_key_name: str, node_label: str, cols: ArrayLike):
    template = f'''UNWIND $recs AS rec\nMERGE(n:{node_label} {{{node_key_name}: rec.{node_key_name}}})'''
    prop_names = [x for x in cols if x != node_key_name]
    if len(prop_names) > 0:
        template = template + '\n' + make_set_clause(prop_names)
    return template + '\nRETURN count(n) AS nodeLoadedCount'


def make_rel_merge_query(source_node_label: str,
                         target_node_label: str,
                         source_node_key_map: Tuple[str, str],
                         target_node_key_map: Tuple[str, str],
                         rel_type: str,
                         cols: ArrayLike,
                         rel_key: str = None):
    merge_statement = f'MERGE(s)-[r:{rel_type}]->(t)'
    if rel_key is not None:
        merge_statement = f'MERGE(s)-[r:{rel_type} {{{rel_key}: rec.{rel_key}}}]->(t)'

    template = f'''\tUNWIND $recs AS rec
    MATCH(s:{source_node_label} {{{source_node_key_map[0]}: rec.{source_node_key_map[1]}}})
    MATCH(t:{target_node_label} {{{target_node_key_map[0]}: rec.{target_node_key_map[1]}}})\n\t''' + merge_statement
    prop_names = [x for x in cols if x not in [rel_key, source_node_key_map[1], target_node_key_map[1]]]
    if len(prop_names) > 0:
        template = template + '\n\t' + make_set_clause(prop_names, 'r')
    return template + '\n\tRETURN count(r) AS relLoadedCount'


def chunks(xs, n=50_000):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def load_nodes(gds: GraphDataScience, node_df: pd.DataFrame, node_key_col: str, node_label: str, chunk_size=50_000):
    records = node_df.to_dict('records')
    print(f'======  loading {node_label} nodes  ======')
    total = len(records)
    print(f'staging {total:,} records')
    query = make_node_merge_query(node_key_col, node_label, node_df.columns.copy())
    cumulative_count = 0
    for recs in chunks(records, chunk_size):
        res = gds.run_cypher(query, params={'recs': recs})
        cumulative_count += res.iloc[0, 0]
        print(f'Loaded {cumulative_count:,} of {total:,} nodes')


def load_rels(gds: GraphDataScience,
              rel_df: pd.DataFrame,
              source_node_label: str,
              target_node_label: str,
              source_node_key_map: Tuple[str, str],
              target_node_key_map: Tuple[str, str],
              rel_type: str,
              rel_key: str = None,
              chunk_size=50_000):
    records = rel_df.to_dict('records')
    print(f'======  loading {rel_type} relationships  ======')
    total = len(records)
    print(f'staging {total:,} records')
    query = make_rel_merge_query(source_node_label, target_node_label, source_node_key_map,
                                 target_node_key_map, rel_type, rel_df.columns.copy(), rel_key)
    cumulative_count = 0
    for recs in chunks(records, chunk_size):
        res = gds.run_cypher(query, params={'recs': recs})
        cumulative_count += res.iloc[0, 0]
        print(f'Loaded {cumulative_count:,} of {total:,} relationships')


def get_ogbn_arxiv_data(map_txt_properties: bool = False, root: str = 'dataset/',
                        write_csv_file: bool = False, csv_file_path: str = '',
                        node_id_col_name="nodeId") -> [DataFrame, DataFrame]:
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
    data, subject = dataset[0]

    citation_df = pd.DataFrame(data['edge_index'].T, columns=['paper', 'citedPaper'])

    node_df = pd.DataFrame(np.union1d(data['edge_index'][0, :], data['edge_index'][1, :]), columns=[node_id_col_name])
    node_df['textEmbedding'] = data['node_feat'].tolist()
    node_df['year'] = data['node_year']
    node_df['subjectId'] = subject

    if map_txt_properties:
        # get raw title and abstract text
        paper_text_df = pd.read_csv('https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv',
                                    sep='\t',
                                    header=None,
                                    names=['paperId', 'title', 'abstract'],
                                    dtype={'paperId': np.int64})

        # get mapping from node id to paper id
        paper_mapping_df = pd.read_csv(f'{root}ogbn_arxiv/mapping/nodeidx2paperid.csv.gz',
                                       names=[node_id_col_name, 'paperId'], header=0)

        # get mapping from subject numeric id to string label
        subject_mapping_df = pd.read_csv(f'{root}ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz',
                                         names=['subjectId', 'subjectLabel'], header=0)

        # merge data together
        paper_df = pd.merge(paper_text_df, paper_mapping_df, on='paperId')
        paper_df = pd.merge(paper_df, node_df, on=node_id_col_name)
        paper_df = pd.merge(paper_df, subject_mapping_df, on='subjectId')
    else:
        paper_df = node_df

    if write_csv_file:
        paper_df.to_csv(csv_file_path + 'papers.csv', index=False)
        citation_df.to_csv(csv_file_path + 'citations.csv', index=False)

    return paper_df, citation_df
