import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if(len(root.findall('object')) == 1):
            #print(len(root.findall('object')))
            for member in root.findall('object'):
                if(member[0].text == 'bird' or member[0].text == 'cat' or member[0].text == 'cow' or
                member[0].text == 'dog' or member[0].text == 'horse' or member[0].text == 'sheep'):
                    #print(member[0].text)
                    value = (root.find('filename').text,
                            int(root.find('size')[0].text),
                            int(root.find('size')[1].text),
                            member[0].text,
                            float(member.find('bndbox')[0].text),
                            float(member.find('bndbox')[1].text),
                            float(member.find('bndbox')[2].text),
                            float(member.find('bndbox')[3].text)
                            )
                    xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
