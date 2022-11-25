import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path, str):
    xml_list = []
    count = 0
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object') and count < 150 :
            if(member[0].text == str):    
                count = count + 1
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
        xml_df = xml_to_csv(image_path, 'bird')
        xml_df = xml_to_csv(image_path, 'cat')
        xml_df = xml_to_csv(image_path, 'cow')
        xml_df = xml_to_csv(image_path, 'dog')
        xml_df = xml_to_csv(image_path, 'horse')
        xml_df = xml_to_csv(image_path, 'sheep')
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
