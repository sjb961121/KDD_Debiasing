import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

def read_item_list():
    file='../data/underexpose_item_feat.csv'
    print('reading item file: ' + file + ' ...')
    for line in open(file, encoding='utf-8').readlines():
        item_index = int(line.strip().split('[')[0][:-1])
        item_embedding_list_text_old = line.strip().split('[')[1][:-2]
        item_embedding_list_image_old= line.strip().split('[')[2][:-1]

        item_embedding_list_tran_text=item_embedding_list_text_old.split(',')
        item_embedding_list_text=list(map(float,item_embedding_list_tran_text))
        item_embedding_list_tran_image = item_embedding_list_image_old.split(',')
        item_embedding_list_image=list(map(float,item_embedding_list_tran_image))

        item_embedding_list_text.extend(item_embedding_list_image)
        item_index_list[item_index]=item_embedding_list_text
    print('finish reading item file...')
    return item_index_list

def read_user_list():
    file='../data/underexpose_user_feat.csv'
    print('reading user file: ' + file + ' ...')
    for line in open(file, encoding='utf-8').readlines():
        user_embedding=[]
        user_index=int(line.strip().split(',')[0])
        age_level=line.strip().split(',')[1]
        if age_level=='':
            age_level=0
        age_level=int(age_level)
        age=to_categorical(age_level,num_classes=10)
        sex_level=line.strip().split(',')[2]
        if sex_level=='F':
            sex=np.array([1,0],dtype='float32')
        else:
            sex=np.array([0,1],dtype='float32')
        city_level=line.strip().split(',')[3]
        if city_level=='':
            city_level='0'
        city=to_categorical(int(city_level),num_classes=10)
        user_embedding.extend(age)
        user_embedding.extend(sex)
        user_embedding.extend(city)
        user_index_list[user_index]=user_embedding
    print('finish reading user file...')
    return user_index_list

def convert_rating(item_index_list):
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    for c in range(7):
        file = test_path+'/underexpose_test_click-{}.csv'.format(c)
        # file = train_path + '/underexpose_train_click-{}.csv'.format(c)
        print('reading rating file ...'+file)
        # item_set = set(item_index_old2new.values())
        # user_pos_ratings = dict()
        # user_neg_ratings = dict()

        for line in open(file, encoding='utf-8').readlines():
            array = line.strip().split(',')

            item_index = int(array[1])
            if item_index not in item_index_list.keys():  # the item is not in the final item set
                continue
            # item_index = item_index_old2new[item_index_old]
            user_index = int(array[0])

            rating = float(array[2])
            # if rating >= 0:
            if user_index not in user_pos_ratings:
                user_pos_ratings[user_index] = set()
            # if len(user_pos_ratings[user_index])<=3:
            user_pos_ratings[user_index].add(item_index)
            # else:
            #     user_pos_ratings[user_index].add(item_index)
            #     user_pos_ratings[user_index].pop()
            # else:
            #     if user_index not in user_neg_ratings:
            #         user_neg_ratings[user_index] = set()
            #     user_neg_ratings[user_index].add(item_index)

        print('converting rating file ...')
        # writer = open('../data/'+'/ratings_train_{}.txt'.format(c), 'w', encoding='utf-8')
        writer = open('../data/' + '/ratings_test_{}.txt'.format(c), 'w', encoding='utf-8')
        item_set = set(item_index_list.keys())
        for user_index, pos_item_set in tqdm(user_pos_ratings.items()):

            for item in pos_item_set:
                item_embedding=item_index_list[item]
                writer.write('%d\t%d\t1\t' % (user_index, item))
                i=0
                for each in item_embedding:
                    if i==255:
                        writer.write('%f'%(each))
                    else:
                        writer.write('%f\t'% (each))
                    i+=1
                writer.write('\n')
            # unwatched_set = item_set - pos_item_set
            # # if user_index in user_neg_ratings:
            # #     unwatched_set -= user_neg_ratings[user_index]
            # for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            #     item_embedding = item_index_list[item]
            #     writer.write('%d\t%d\t0\t' % (user_index, item))
            #     i=0
            #     for each in item_embedding:
            #         if i == 255:
            #             writer.write('%f' % (each))
            #         else:
            #             writer.write('%f\t' % (each))
            #         i += 1
            #     writer.write('\n')
        writer.close()
        print('number of users: %d' % len(user_pos_ratings.keys()))
        print('number of items: %d' % len(item_set))


train_path = '../data/underexpose_train'
test_path = '../data/underexpose_test'
item_index_list=dict()
user_index_list=dict()

def load_data():
    return read_item_list()

if __name__ == '__main__':
    read_item_list()
    # read_user_list()
    convert_rating(item_index_list)