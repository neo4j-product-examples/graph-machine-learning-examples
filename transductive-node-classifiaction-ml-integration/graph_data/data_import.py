from ogb.nodeproppred import NodePropPredDataset
import pandas as pd
import numpy as np
from pandas import DataFrame


def get_ogbn_arxiv_data(map_txt_properties: bool = False, root: str = 'dataset/',
                          write_csv_file: bool = False, csv_file_path: str = '') -> [DataFrame, DataFrame]:
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
    data, subject = dataset[0]

    citation_df = pd.DataFrame(data['edge_index'].T, columns=['paper', 'citedPaper'])

    node_df = pd.DataFrame(np.union1d(data['edge_index'][0, :], data['edge_index'][1, :]), columns=['nodeId'])
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
        paper_mapping_df = pd.read_csv(f'{root}ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', names=['nodeId', 'paperId'],
                                       header=0)

        # get mapping from subject numeric id to string label
        subject_mapping_df = pd.read_csv(f'{root}ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz',
                                         names=['subjectId', 'subjectLabel'], header=0)

        # merge data together
        paper_df = pd.merge(paper_text_df, paper_mapping_df, on='paperId')
        paper_df = pd.merge(paper_df, node_df, on='nodeId')
        paper_df = pd.merge(paper_df, subject_mapping_df, on='subjectId')
    else:
        paper_df = node_df

    if write_csv_file:
        paper_df.to_csv(csv_file_path + 'papers.csv', index=False)
        citation_df.to_csv(csv_file_path + 'citations.csv', index=False)

    return paper_df, citation_df