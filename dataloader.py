import os
from lxml import etree

def extract_events(folder_path):
    events = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tml'):
            file_path = os.path.join(folder_path, filename)
            tree = etree.parse(file_path)
            root = tree.getroot()
            for event in root.findall('.//EVENT'):
                events.append({
                    'file': filename,
                    'eid': event.get('eid'),
                    'text': event.text,
                    'class': event.get('class')
                })
    return events

folder_path = 'TE3-Silver-data'
events = extract_events(folder_path)
print(events[:10])

