import requests, secrets, uuid
from IPython.display import display
import ipywidgets as w
from google.colab import output
output.enable_custom_widget_manager()
root='http://127.0.0.1:8000/api/v1'
creds={'email':f'colab-{uuid.uuid4().hex}@example.com','password':secrets.token_urlsafe(20)}
s=requests.Session()
r=s.post(root+'/auth/register',json=creds); r.raise_for_status()
r=s.post(root+'/auth/login',json=creds); r.raise_for_status()
s.headers['Authorization']='Bearer '+r.json()['access_token']
r=s.post(root+'/knowledge-bases',json={'name':'Мои документы','description':''}); r.raise_for_status()
base_id=r.json()['id']
upload=w.FileUpload(accept='.pdf,.docx,.txt,.md',multiple=False,description='Документ')
index_button=w.Button(description='Загрузить',button_style='primary')
status_button=w.Button(description='Обновить статус')
query=w.Text(description='Поиск:',placeholder='Введите запрос',layout=w.Layout(width='80%'))
search_button=w.Button(description='Найти',button_style='success')
results=w.Output()
def show_docs(_=None):
    with results:
        results.clear_output()
        try:
            r=s.get(root+f'/knowledge-bases/{base_id}/documents'); r.raise_for_status()
            for d in r.json(): print(d['filename'], '—', d['status'], d.get('error') or '')
        except Exception as e: print('Ошибка:',e)
def add_file(_):
    with results:
        results.clear_output()
        try:
            if not upload.value: print('Выберите документ.'); return
            f=next(iter(upload.value.values())) if isinstance(upload.value,dict) else upload.value[0]
            r=s.post(root+f'/knowledge-bases/{base_id}/documents',files={'file':(f['name'],bytes(f['content']))},timeout=60); r.raise_for_status()
            print('Документ загружен. Дождитесь индексации и нажмите «Обновить статус».')
        except Exception as e: print('Ошибка:',e, getattr(locals().get('r'), 'text', ''))
def do_search(_):
    with results:
        results.clear_output()
        try:
            r=s.post(root+'/search',json={'knowledge_base_id':base_id,'query':query.value,'strategy':'keyword','limit':5},timeout=120); r.raise_for_status()
            hits=r.json()['results']
            if not hits: print('Совпадений нет. Убедитесь, что документ обработан.')
            for i,h in enumerate(hits,1): print(f"{i}. Страница {h.get('page') or '—'}\n{h['text']}\n")
        except Exception as e: print('Ошибка:',e, getattr(locals().get('r'), 'text', ''))
index_button.on_click(add_file); status_button.on_click(show_docs); search_button.on_click(do_search)
display(w.VBox([w.HTML('<h3>AI Knowledge Engine: документы и поиск</h3><p>Данные хранятся только в этой сессии Colab.</p>'),w.HBox([upload,index_button,status_button]),w.HBox([query,search_button]),results]))
