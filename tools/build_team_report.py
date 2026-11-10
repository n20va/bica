"""Build shareable pilot PDF/PNG/XLSX from the preserved real Colab archive.

Requires matplotlib and openpyxl. Does not run a model or change raw results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/bica_report_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
MODES = ['direct', 'repeat', 'hypotheses', 'evidence']
NAMES = ['Одиночный запрос', 'Повторная проверка', 'Поиск альтернатив', 'Альтернативы + признаки']
STEPS = [1, 3, 3, 3]
NAVY, MUTED, TEAL, CORAL, BLUE = '#17243A', '#63738A', '#168F81', '#CC644B', '#5877C9'
BG, BORDER = '#F3F6FB', '#E0E7F0'


def read_data(path):
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read('pilot_v1/config.json'))
        summary = json.loads(z.read('pilot_v1/summary.json'))
        rows = [json.loads(s) for s in z.read('pilot_v1/results.jsonl').decode().splitlines()]
        events = [json.loads(s) for s in z.read('pilot_v1/trace.jsonl').decode().splitlines()]
    responses = [e for e in events if e['event'] == 'response']
    assert not summary['synthetic'] and len(rows) == summary['completed_cases']
    assert sum(e['event'] == 'request' for e in events) == summary['attempted_calls']
    assert sum(r['status'] == 'error' for r in rows) == summary['errors']
    stats = []
    for mode, name, steps in zip(MODES, NAMES, STEPS):
        subset = [r for r in rows if r['mode'] == mode]
        ids = {r['case_id'] for r in subset}
        calls = [e for e in responses if e['case_id'] in ids]
        ok = sum(r['status'] == 'ok' for r in subset)
        stats.append(dict(mode=mode, name=name, cases=len(subset), ok=ok,
                          errors=len(subset)-ok, calls=len(calls), planned=len(subset)*steps,
                          seconds=sum(e['elapsed_s'] for e in calls)))
    image_ids = {r['image_id'] for r in rows}
    same = sum(next(r for r in rows if r['image_id'] == i and r['mode'] == 'direct')['answer'] ==
               next(r for r in rows if r['image_id'] == i and r['mode'] == 'repeat')['answer']
               for i in image_ids)
    return config, summary, rows, stats, same


def visual_report(out, summary, stats, same):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': NAVY, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': NAVY, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(13.33, 9.4), facecolor=BG)
    def text(x, y, value, size=11, color=NAVY, weight='normal', **kw):
        fig.text(x, y, value, fontsize=size, color=color, weight=weight, **kw)
    def panel(x, y, w, h, color='white'):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                     facecolor=color, edgecolor=BORDER, linewidth=0.7,
                                     transform=fig.transFigure, zorder=0))
    text(.045, .952, 'BICA  /  ОТЧЁТ КОМАНДЕ', 10, TEAL, 'bold')
    text(.955, .952, '10 НОЯБРЯ 2026  ·  ПИЛОТ 01', 9, MUTED, ha='right')
    text(.045, .906, 'Как VLM воспринимает неоднозначные изображения', 24, weight='bold')
    text(.045, .872, 'Qwen2.5-VL-3B-Instruct  •  Google Colab / Tesla T4  •  только нейтральные запросы', 11, MUTED)

    cards = [('6', 'изображений AmbiBench'), ('24', 'случая в четырёх режимах'),
             (str(summary['attempted_calls']), 'реальных обращений к модели'),
             (f"{summary['completed_cases']-summary['errors']} / {summary['planned_cases']}", 'без технической ошибки')]
    for i, (value, label) in enumerate(cards):
        x = .045+i*.233
        panel(x, .757, .211, .088)
        text(x+.016, .799, value, 27, TEAL if i==3 else NAVY, 'bold')
        text(x+.016, .773, label, 9, MUTED)

    panel(.045, .415, .443, .308)
    panel(.512, .415, .443, .308)
    text(.063, .690, 'Технический статус', 14, weight='bold')
    text(.063, .666, 'Число случаев; каждый режим проверялся на 6 картинках', 9, MUTED)
    ax = fig.add_axes([.224, .468, .236, .176], facecolor='white')
    y = list(range(4))
    ax.barh(y, [s['ok'] for s in stats], color=TEAL, height=.52)
    ax.barh(y, [s['errors'] for s in stats], left=[s['ok'] for s in stats], color=CORAL, height=.52)
    for i, s in enumerate(stats):
        ax.text(3, i, f"{s['ok']} принято" if s['ok'] else f"{s['errors']} с ошибкой",
                ha='center', va='center', color='white', fontsize=9, weight='bold')
    ax.set_yticks(y, ['Одиночный запрос', 'Повторная проверка', 'Поиск альтернатив', 'Альтернативы + признаки'], fontsize=9)
    ax.set_xlim(0, 6); ax.set_xticks([0, 2, 4, 6]); ax.invert_yaxis()
    ax.tick_params(axis='both', length=0, pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(handles=[Patch(color=TEAL, label='Ответ принят по формату'), Patch(color=CORAL, label='Ошибка')],
              loc='lower right', bbox_to_anchor=(1, -0.35), ncol=2, frameon=False, fontsize=8)

    text(.530, .690, 'Стоимость в обращениях', 14, weight='bold')
    text(.530, .666, 'Фактические вызовы / план при прохождении всех шагов', 9, MUTED)
    ax2 = fig.add_axes([.556, .488, .359, .153], facecolor='white')
    ax2.bar(y, [s['planned'] for s in stats], color='#E4E9F5', width=.55, label='План')
    ax2.bar(y, [s['calls'] for s in stats], color=BLUE, width=.55, label='Выполнено')
    for i, s in enumerate(stats):
        ax2.text(i, s['planned']+.7, f"{s['calls']} / {s['planned']}", ha='center', fontsize=10, weight='bold')
    ax2.set_xticks(y, ['Одиночный', 'Повтор', 'Альтернативы', '+ признаки'], fontsize=8)
    ax2.set_ylim(0, 22); ax2.set_yticks([0, 6, 12, 18]); ax2.set_axisbelow(True)
    ax2.grid(axis='y', color=BORDER, linewidth=.6)
    ax2.tick_params(axis='both', length=0, pad=6)
    for spine in ax2.spines.values(): spine.set_visible(False)
    text(.530, .438, 'Меньше вызовов в двух режимах — следствие остановки с ошибкой.', 8.5, MUTED)

    text(.045, .381, 'Сводная таблица', 13, weight='bold')
    ax3 = fig.add_axes([.045, .204, .910, .156]); ax3.axis('off')
    table_rows = [[s['name'], s['cases'], s['ok'], s['errors'], f"{s['ok']/s['cases']:.0%}",
                   s['calls'], f"{s['seconds']:.1f}".replace('.', ',')] for s in stats]
    table = ax3.table(cellText=table_rows,
                      colLabels=['Режим', 'Случаев', 'Принято', 'Ошибок', 'Принято, %', 'Вызовов', 'Время, с'],
                      colWidths=[.34,.09,.10,.10,.13,.11,.13], cellLoc='center', bbox=[0,0,1,1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r,c), cell in table.get_celld().items():
        cell.set_linewidth(.6); cell.set_edgecolor(BORDER)
        cell.set_facecolor(NAVY if r == 0 else ('white' if r%2 else '#EBF0F8'))
        cell.get_text().set_color('white' if r == 0 else NAVY)
        if r == 0: cell.get_text().set_weight('bold')
        if c == 0: cell.get_text().set_ha('left'); cell.PAD=.04

    text(.045, .181, 'Принято = ответ прошёл проверку формата. Время — сумма обращений, без загрузки весов.', 9, MUTED)
    panel(.045, .066, .910, .091, '#E8F3F0')
    text(.061, .132, f'НАБЛЮДЕНИЕ  ·  {same}/6 ответов совпали после повторной проверки', 11, TEAL, 'bold')
    text(.061, .108, 'Все 12 ошибок: на шаге поиска альтернатив получено 5–6 гипотез при лимите 4.', 10)
    text(.061, .084, 'Далее: проверить уточнённую инструкцию, сверить эталоны и выполнить независимую разметку.', 10)
    text(.045, .035, 'Это отладочный пилот. Точность распознавания и преимущество агента пока не установлены.', 10, weight='bold')
    text(.045, .015, 'Источник: bica_real_pilot_results.zip / pilot_v1  •  6 изображений  •  код и промпты зафиксированы в журнале', 8, MUTED)
    fig.savefig(out/'BICA_team_report_2026-11-10.pdf', facecolor=BG,
                metadata={'Title':'BICA — первый реальный пилот', 'Author':'BICA project team'})
    fig.savefig(out/'BICA_team_report_2026-11-10.png', dpi=180, facecolor=BG)
    plt.close(fig)


def workbook_report(out, source, config, summary, rows, stats, same):
    wb = Workbook(); ws = wb.active; ws.title = 'Сводка'
    ws.sheet_view.showGridLines = False
    ws.merge_cells('A1:H1'); ws['A1'] = 'BICA | Визуальная неоднозначность — пилот 10.11.2026'
    ws['A1'].font = Font(name='Calibri', size=20, bold=True, color=NAVY[1:])
    ws.row_dimensions[1].height = 36
    ws.merge_cells('A2:H2'); ws['A2'] = 'Qwen2.5-VL-3B-Instruct • Tesla T4 • 6 изображений • только нейтральное условие'
    ws.merge_cells('A3:H3'); ws['A3'] = '24 завершённых случая • 48 вызовов • 12 ответов принято по формату • 12 ошибок'
    ws.merge_cells('A4:H4'); ws['A4'] = 'ВАЖНО: техническая успешность не является точностью распознавания.'
    ws['A4'].font = Font(bold=True, color=CORAL[1:]); ws.row_dimensions[4].height = 26
    headers = ['Режим', 'Случаев', 'Принято', 'Ошибок', 'Принято, %', 'Вызовов', 'План вызовов', 'Время, с']
    for j,h in enumerate(headers,1): ws.cell(6,j,h)
    for r,s in enumerate(stats,7):
        values = [s['name'], s['cases'], s['ok'], s['errors'], s['ok']/s['cases'], s['calls'], s['planned'], round(s['seconds'],2)]
        for j,v in enumerate(values,1): ws.cell(r,j,v)
        ws.cell(r,5).number_format='0%'; ws.cell(r,8).number_format='0.00'
    for j,v in enumerate(['Итого',24,12,12,.5,48,60,round(sum(s['seconds'] for s in stats),2)],1): ws.cell(11,j,v)
    ws.cell(11,5).number_format='0%'
    ws.merge_cells('A13:H13'); ws['A13'] = f'Наблюдение: итоговые ответы direct и repeat совпали на {same} из 6 изображений.'
    ws.merge_cells('A14:H14'); ws['A14'] = 'Все 12 ошибок: 5–6 гипотез вместо максимум 4. Финальный шаг двух режимов не выполнен.'
    ws.merge_cells('A15:H15'); ws['A15'] = 'Время = сумма длительностей обращений; без загрузки модели. Меньше вызовов ≠ выше эффективность.'

    c1=BarChart(); c1.type='bar'; c1.grouping='stacked'; c1.overlap=100
    c1.title='Технический статус по режимам'; c1.x_axis.title='Количество случаев'
    c1.add_data(Reference(ws,min_col=3,max_col=4,min_row=6,max_row=10),titles_from_data=True)
    c1.set_categories(Reference(ws,min_col=1,min_row=7,max_row=10))
    c1.x_axis.scaling.min=0; c1.x_axis.scaling.max=6; c1.x_axis.majorUnit=2
    for series,color in zip(c1.series,[TEAL,CORAL]):
        series.graphicalProperties.solidFill=color[1:]; series.graphicalProperties.line.noFill=True
    c1.height=10; c1.width=19; c1.legend.position='b'; ws.add_chart(c1,'A18')
    c2=BarChart(); c2.type='col'; c2.grouping='clustered'
    c2.title='Вызовы: факт и план'; c2.y_axis.title='Количество обращений'
    c2.add_data(Reference(ws,min_col=6,max_col=7,min_row=6,max_row=10),titles_from_data=True)
    c2.set_categories(Reference(ws,min_col=1,min_row=7,max_row=10))
    for series,color in zip(c2.series,[BLUE,'#CBD4E6']):
        series.graphicalProperties.solidFill=color[1:]; series.graphicalProperties.line.noFill=True
    c2.y_axis.scaling.min=0; c2.y_axis.scaling.max=20; c2.y_axis.majorUnit=5
    c2.height=10; c2.width=19; c2.legend.position='b'; ws.add_chart(c2,'A39')
    ws.freeze_panes='B7'
    for col in range(1,9): ws.column_dimensions[get_column_letter(col)].width=16 if col>1 else 35
    for row in ws.iter_rows(min_row=6,max_row=11):
        for cell in row:
            header = cell.row in [6,11]
            cell.fill=PatternFill('solid',fgColor=NAVY[1:] if header else ('FFFFFF' if cell.row%2 else 'EBF0F8'))
            cell.font=Font(name='Calibri',size=11,bold=header,color='FFFFFF' if header else NAVY[1:])
            cell.alignment=Alignment(vertical='center',horizontal='left' if cell.column==1 else 'center',wrap_text=True)
        ws.row_dimensions[row[0].row].height=30
    ws.print_options.horizontalCentered=True
    ws.sheet_properties.pageSetUpPr.fitToPage=True
    ws.page_setup.orientation='landscape'; ws.page_setup.paperSize=ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth=1; ws.page_setup.fitToHeight=1; ws.print_area='A1:H59'

    details=wb.create_sheet('24 случая')
    details.append(['Изображение','Режим','Статус','Причина ошибки','Интерпретации модели'])
    for r in sorted(rows,key=lambda r:(r['image_id'], MODES.index(r['mode']))):
        details.append([r['image_id'], NAMES[MODES.index(r['mode'])], 'Принято по формату' if r['status']=='ok' else 'Ошибка',
                        r.get('detail',''), '\n'.join(x['label'] for x in r.get('answer',{}).get('interpretations',[]))])
    details.freeze_panes='C2'; details.auto_filter.ref=details.dimensions
    for col,width in zip('ABCDE',[23,32,25,48,65]): details.column_dimensions[col].width=width
    for row in details.iter_rows(min_row=2): details.row_dimensions[row[0].row].height=58

    info=wb.create_sheet('Методика и источник')
    notes = [
        ['Параметр','Значение'], ['Модель',config['client']['model']], ['Ревизия весов',config['client']['revision']],
        ['GPU',config['client']['gpu']], ['Декодирование','greedy; seed 42; max_tokens 768; float16; SDPA'],
        ['Источник',source.name+' / pilot_v1'], ['SHA-256 архива',hashlib.sha256(source.read_bytes()).hexdigest()],
        ['Дата запуска','10.11.2026'], ['Выборка','6 отладочных изображений AmbiBench; только c0, без подсказок'],
        ['Единица наблюдения','Один случай = изображение × режим. 24 случая не являются 24 независимыми изображениями.'],
        ['Принято по формату','Парсер принял структуру итогового ответа. Это не подтверждение правильности интерпретаций.'],
        ['Завершено','Результат сохранён с успехом или ошибкой. Все 24 завершены, успешны технически только 12.'],
        ['Число вызовов','48 вместо 60: двенадцать случаев остановились до финального шага.'],
        ['Время','Сумма elapsed_s у событий response. Не включает загрузку модели и подготовку.'],
        ['Совпадение ответов',f'Итоговые структурированные ответы direct и repeat равны на {same}/6 изображениях.'],
        ['Точность','Не вычислена. ratings.csv пуст; нужны независимые человеческие оценки и проверка эталонов.'],
        ['Причина ошибок','На alternatives модель выдала 5–6 гипотез при лимите 4. Ни один ответ не достиг лимита токенов.'],
        ['Ограничения','Малая отладочная выборка, одна модель, нет обычных контрольных изображений или ложных подсказок.'],
        ['Следующий шаг','Подготовлена инструкция alternatives v3. Её успех на GPU ещё не подтверждён.'],
        ['Научный вывод','Преимущество агента не установлено. Отчёт описывает инфраструктуру и предварительные наблюдения.'],
    ]
    for row in notes: info.append(row)
    info.column_dimensions['A'].width=28; info.column_dimensions['B'].width=115
    info.freeze_panes='B2'
    for row in info.iter_rows(min_row=2): info.row_dimensions[row[0].row].height=42
    for sheet in [details,info]:
        sheet.sheet_view.showGridLines=False
        for row in sheet:
            for cell in row:
                cell.font=Font(name='Calibri',size=11,bold=cell.row==1,color='FFFFFF' if cell.row==1 else NAVY[1:])
                cell.fill=PatternFill('solid',fgColor=NAVY[1:] if cell.row==1 else ('FFFFFF' if cell.row%2 else 'EBF0F8'))
                cell.alignment=Alignment(vertical='center',wrap_text=True)
        sheet.row_dimensions[1].height=30
    path=out/'BICA_team_report_2026-11-10.xlsx'; wb.save(path)
    check=load_workbook(path)
    assert len(check['Сводка']._charts)==2 and check['24 случая'].max_row==25
    assert check['Сводка']['F11'].value==summary['attempted_calls']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=ROOT/'runs/imported/ae999bb543bf/bica_real_pilot_results.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'deliverables/team_report_2026-11-10')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    config,summary,rows,stats,same=read_data(args.archive)
    visual_report(args.output,summary,stats,same)
    workbook_report(args.output,args.archive,config,summary,rows,stats,same)
    (args.output/'report_data.json').write_text(json.dumps({'source_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'summary':summary,'by_mode':stats,'direct_repeat_equal':same},ensure_ascii=False,indent=2))
    for path in sorted(args.output.iterdir()): print(path.name,path.stat().st_size,'bytes')


if __name__=='__main__': main()
