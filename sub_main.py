import POS_Embed
import POS_Tagging
import data_processor
import Combined_POS_Processor

def saveFile():
    pos_embed = POS_Embed.POS_Embedder()
    # pos_embed.model(200)
    # pos_embed.model_continue(200)
    embeddings = pos_embed.get_embedding()
    print(embeddings.shape)

    fileName = 'C:\\Users\\Administrator\\Desktop\\qadataset\\pos_embed'
    f = open(fileName, 'w')

    for a in range(84):
        string = ""

        for i in range(128):
            string = string + str(embeddings[a, 0, i]) + " "
        string = string + "#"

        f.write(string)

    f.close()

pos_Embed = POS_Embed.POS_Embedder()
dataholder = data_processor.Data_holder()
dataholder.set_batch()
pos_Embed.set_Data_Hlder(dataholder)

"""
pos_Modeler = Combined_POS_Processor.POS_Processor(is_Setting_Embedder=False)
pos_Modeler.setEmbedder(pos_Embed)

pos_Embed.set_POS_Tagger(pos_Modeler)

a, b = pos_Embed.read_POS_From_Embed()
pos_Embed.get_Fianl_POS(dataholder.paragraph_arr, dataholder.numberOf_available_question, a)
"""

print('Complete')
