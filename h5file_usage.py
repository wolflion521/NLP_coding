# 保存
if os.path.exists(filename):
    os.remove(filename)
with h5py.File(filename) as f:
    out_datas = f.create_group('similarity_data')
    out_datas['origin_feature'] = origin_feature_array
    out_datas['standard_feature'] = standard_feature_array
    out_datas['origin_length_include_CLS_SEP'] = origin_length
    out_datas['standard_length_include_CLS_SEP'] = standard_length
    out_datas['similarity'] = similarity_array
    f['bert_data'] = bert_data
    # bert_data = []
    print(' bert features saved')
# 读取
with h5py.File(file, 'r') as f:
    out_datas = f['similarity_data']
    origin_feature_array = out_datas['origin_feature']
    standard_feature_array = out_datas['standard_feature']
    origin_length = out_datas['origin_length_include_CLS_SEP']
    standard_length = out_datas['standard_length_include_CLS_SEP'] 
    similarity_array = out_datas['similarity'] 
    

h5文件存储的要求有哪些？我之前用dict存放每一笔数据，然后用[]存放所有dict，之后想存成h5.不行。看了别人代码，意识到h5能存储的是np.array.还有json存不了nparray                
                
