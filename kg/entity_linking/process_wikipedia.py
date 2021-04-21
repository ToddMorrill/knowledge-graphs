import csv
import os

ENCODING = 'utf-8'


class ProcessPagesWorker(object):
    def __init__(self, output_queue, config):
        self.output_queue = output_queue
        self.config = config

    def process_template(self, id, title):
        self.output_queue.put({'template': [id, title]})

    def process_article(self, id, title, text):
        self.output_queue.put({'article': [id, title]})

    def process_redirect(self, id, title, redirect):
        self.output_queue.put({'redirect': [id, title, redirect]})

    def report_progress(self, completed):
        self.output_queue.put({'completed': completed})


class ProcessPages(object):
    def __init__(self, output_path):
        self.output_path = output_path

    def open(self):
        path_articles = os.path.join(self.output_path, 'article.csv')
        path_redirect = os.path.join(self.output_path, 'redirect.csv')
        path_template = os.path.join(self.output_path, 'template.csv')

        self.articles_fp = open(path_articles, 'w', encoding=ENCODING)
        self.redirect_fp = open(path_redirect, 'w', encoding=ENCODING)
        self.template_fp = open(path_template, 'w', encoding=ENCODING)

        self.articles_writer = csv.writer(self.articles_fp,
                                         quoting=csv.QUOTE_MINIMAL)
        self.redirect_writer = csv.writer(self.redirect_fp,
                                         quoting=csv.QUOTE_MINIMAL)
        self.template_writer = csv.writer(self.template_fp,
                                         quoting=csv.QUOTE_MINIMAL)

        self.articles_writer.writerow(['id', 'title'])
        self.redirect_writer.writerow(['id', 'title', 'redirect'])
        self.template_writer.writerow(['id', 'title'])

    def close(self):
        self.articles_fp.close()
        self.redirect_fp.close()
        self.template_fp.close()

    def handle_event(self, evt):

        if 'article' in evt:
            self.articles_writer.writerow(evt['article'])
        elif 'template' in evt:
            self.template_writer.writerow(evt['template'])
        elif 'redirect' in evt:
            self.redirect_writer.writerow(evt['redirect'])

    def get_worker_class(self, output_queue, config):
        return ProcessPagesWorker(output_queue, config)