
#basics
import random
import pandas as pd
import torch
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def get_tree_from_file(self, data_dir):
        list_to_df = []
        train = glob.glob("{}/*/*/*.xml".format(data_dir)) #returns te.
        test = glob.glob("{}/*/*/*/*.xml".format(data_dir))
        # file = [glob.glob("{}*.xml".format(item)) for item in directories] #returns equal many folders containing xml-files
        for item in train, test:
            list_to_df.append(item) #iterate and add to final folder to get format nested list with a list pr. directory containing xml files for furture DF-split.
        flat_list = [i for sublist in list_to_df for i in sublist]
        return flat_list


    def make_pd_from_tree(self, lists_of_file_names):
        # print(lists_of_file_names)
        self.id2word = {}
        self.id2ner = {}
        self.id2ner[0] = 'None'
        punctuation = ['.',',',':',';','!','?','!','/', "'", '%', '(', ')']
        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]) #for test purposes
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) # for test
        df_index = 1
        ner_index = 1
        for filename in lists_of_file_names:
            if 'Test' in filename:
                split = 'test'
            else:
                split = 'train'
            xml = ET.parse(filename)
            root = xml.getroot()
            for sentence in root.iter('sentence'): # get sent id and text 
                sent_id = sentence.get('id')
                sentence_text = sentence.get('text')
                tokens = sentence_text.split(' ')
                current_index = 0
                # print(sentence_text)
                for word in tokens:
                    # print(    word)
                    character_start = current_index
                    character_end = current_index + len(word)-1 #minus 1 because len(word) doesnt take index-0 into account
                    if word:
                        if word[-1] in punctuation:
                            word = word[:-1]
                            character_end -= 1  #for removed punctuation item decreasing the length by 1
                            # print(word)
                        if word.isalpha() == False:
                            pass
                        else:
                            current_index += len(word)+1 # for beginning of next word, as it cannot be on the same slot as the previous ;) 
                            self.data_df.loc[df_index] = [sent_id, word, character_start, character_end, split] #word to be LabelEncoded later
                            df_index += 1
                for item in sentence:
                        # print(item)
                        if item.tag == 'entity': #ensuring that ther e is an entity at all.
                            ner_id = item.get('type') #ner = type
                            # print(ner_id)
                            token_id = item.get('text') #entity = name / text
                            if ";" in item.get('charOffset'):
                                char_offsets = item.get('charOffset').split(';') #split charoffset
                                for span in char_offsets:
                                    char_start_id, char_end_id = span.split('-')
                                    char_start_id, char_end_id = int(char_start_id), int(char_end_id)
                                    # print(char_start_id, char_end_id)
                                    self.ner_df.loc[ner_index] = [sent_id, ner_id, char_start_id, char_end_id]
                                    ner_index +=1
                            else: 
                                char_start_id, char_end_id = item.get('charOffset').split('-')[0], item.get('charOffset').split('-')[1] # split char off set
                                char_start_id, char_end_id = int(char_start_id), int(char_end_id)
                                self.ner_df.loc[ner_index] = [sent_id, ner_id, char_start_id, char_end_id]
                                # print(data_df, ner_df)
                                ner_index += 1     
            #if ner_index > 10: # to get smaller dataset, only for tester. 
             #   break 
        pass


    def data_df_label_encoding(self):
        #label_encoding
        lb_make_df = LabelEncoder()
        self.data_df['token_id'] = lb_make_df.fit_transform(self.data_df['token_id'])
        lb_make_df_name_mapping = dict(zip(lb_make_df.classes_, lb_make_df.transform(lb_make_df.classes_)))
        self.id2word = lb_make_df_name_mapping
        # print(data_df)
        pass


    def ner_id_label_encoding(self):
        #label_encoding
        lb_make_ner = LabelEncoder()
        self.ner_df['ner_id'] = lb_make_ner.fit_transform(self.ner_df['ner_id'])
        lb_make_ner_name_mapping = dict(zip(lb_make_ner.classes_, lb_make_ner.transform(lb_make_ner.classes_)))
        self.id2ner = lb_make_ner_name_mapping
        # print(data_df)
        pass


    def max_length(self): 
    
        longest = self.data_df["sentence_id"].value_counts()
        # print(longest)
        max_sample_length = longest.max() 

        return max_sample_length

    def padding(self, max_length, sentence_to_pad):
        length_1 = len(sentence_to_pad)
        diff = max_length - length_1
        padding = [-1] * diff
        return  sentence_to_pad.extend(padding)





# print(df_to_tens(max_length(data_df), train, ner_df))


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        

        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]) #for test purposes
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) # for test

        # print(self.get_tree_from_file(data_dir))

        # fill out the DF's above.
        # self.make_pd_from_tree(self.get_tree_from_file(data_dir))
        self.make_pd_from_tree(self.get_tree_from_file(data_dir))
        # print(data_frames)
        print('Printing Data_df')
        print(self.data_df)
        print('---')
        print('Printing Ner_df')
        print(self.ner_df)

        #Label-Encode text-values in DF
        print('Encoding labels into numeric values...')
        self.data_df_label_encoding()
        self.ner_id_label_encoding()

        print(self.data_df)
        print(self.ner_df)

        self.vocab = list(self.id2word.keys())

        # id2word = data_df_label_encoding(data_df)[1]
        self.id2word = dict(zip(self.id2word.values(), self.id2word.keys()))
# print(id2word)
        # id2ner = ner_id_label_encoding(ner_df)[1]
        self.id2ner = dict(zip(self.id2ner.values(), self.id2ner.keys()))
        print(self.id2word)
        print(self.id2ner)
        self.max_sample_length = self.max_length()
        # data_df_label_encoding() # [1]
        # ner_id_label_encoding() # [1]

        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.

        pass


    def df_to_tens(self, max_length, df_split):
        """
        The idea here is to convert a DF-split into a list of sorts and loop it rhough the NER to see if any hits.

        """
        y_tensor = []
        merged = df_split.merge(self.ner_df, how='left', left_on=['sentence_id', 'char_start_id', 'char_end_id'], right_on=['sentence_id', 'char_start_id', 'char_end_id']).fillna(-1)

        merged_df = merged.sort_values(by=['sentence_id', 'char_start_id'])
        # print(max_length)
        merged = merged.groupby('sentence_id')
        # print(merged_df)
        for sentence_id, df_grouped_by_sentence in merged:
            df_sent = [int(v) for v in list(df_grouped_by_sentence['ner_id'])]
            # print(df_sent)
            if len(df_sent) < self.max_sample_length: #pad
                self.padding(self.max_sample_length, df_sent)
                y_tensor.append(df_sent)
            else:
                if len(df_sent) >= self.max_sample_length:
                    y_tensor.append(df_sent)
        y_tensor = np.asarray(y_tensor, dtype=np.float32)
                    
        return y_tensor



    def return_tensor_data(self, tensor):
        """
        to return a short list of count-values
        """
        dic = dict()
        array = np.array(tensor.detach().cpu())

        types, counts = np.unique(array, return_counts=True) #returns tuple of (x,y)
        types, counts = types.astype(int), counts.astype(int)

        dic = dict(zip(types, counts))

        if -1.0 in dic:
            del dic[-1.0]


        

        return dic





    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU

        # print('Making train and validation set in ratio 8-2 of total training set . . . ')
        # print('Making test-set . . . ')
        self.train, self.validate= np.split(self.data_df.loc[self.data_df['split'] == 'train'].sample(frac=1), [int(.2*len(self.data_df))])
        self.test = self.data_df.loc[self.data_df['split'] == 'test']

        # print(train)
        # print('---')
        # print(validate)
        # print('---')
        # print(test)

        # print('turning into np arrays . . . ')
        nparray_train = self.df_to_tens(self.max_sample_length, self.train)

        nparray_validate = self.df_to_tens(self.max_sample_length, self.validate)
        nparray_test = self.df_to_tens(self.max_sample_length, self.test)

        # print(type(nparray_train))
        # print('---')
        # print(nparray_validate)
        # print('---')
        # print(nparray_test)

        # print('turning np arrays into tensor objects.')
        tensor_train = torch.from_numpy(nparray_train).to(self.device)
        tensor_validate = torch.from_numpy(nparray_validate).to(self.device)
        tensor_test = torch.from_numpy(nparray_test).to(self.device)

        # print(type(tensor_train))
        # print('done . . .')



        return tensor_train, tensor_validate, tensor_test

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        train = self.return_tensor_data(self.get_y()[0])
        validate = self.return_tensor_data(self.get_y()[1])
        test = self.return_tensor_data(self.get_y()[2])
        # print(train)
        # print(validate)

        df = pd.DataFrame([train, validate, test], index=['train', 'validate', 'test']) 

        # print(df)

        fig, ax= plt.subplots(1,1, figsize=(6,5))

        df.plot.bar(ax=ax)

        fig.tight_layout()
        fig.show()

        ax.set_xlabel('Df Split')


        plt.show()


        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
            def get_list(df_split):
                nr_of_dist = []
                train_split = df_split.sort_values(by='sentence_id')
                train_split = train_split.groupby('sentence_id')
                for name, items in train_split:
                    nr_of_dist.append(len(items))

                nr_of_dist = sorted(nr_of_dist)
                return nr_of_dist
            
            
            train_sample_length = get_list(self.train)
            validate_sample_length = get_list(self.validate)
            test_sample_length = get_list(self.test)
            # print(validate_sample_length)



            fig, ax = plt.subplots(tight_layout=True)
            ax.hist(train_sample_length, density=True, bins=20, label = 'Train')
            ax.hist(validate_sample_length, density=True, bins=30, label = 'Validate')
            ax.hist(test_sample_length, density=True, bins=40, label = 'Test')

            fig.show()
            plt.legend()
            plt.show()
            pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on

            def get_no():
                counter_dict = dict()
                no = []
                df_sort = self.ner_df.sort_values(by='sentence_id')
                df_group = df_sort.groupby('sentence_id')
                for name, item in df_group:
                    no.append(item['ner_id'].values)


                lst = [l.tolist() for l in no]
                

                for item in lst:
                    for i in item: 
                        if i not in counter_dict:
                            counter_dict[i] = 1
                        else:
                            counter_dict[i] += 1

                





                return counter_dict
            print('Made over iterations of NER_DF and not individual split_DF')
            d = get_no()
            plt.bar(range(len(d)), list(d.values()), align='center')
            plt.xticks(range(len(d)), list(d.keys()))
            plt.ylabel('Number of sentences')
            plt.xlabel('NER-number')
            plt.show()
        
            pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass

