import fitz

# pdfs = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' ]
pdfs = 'pdfs/RCTG'

# for pdf in pdfs:
for i in range(1, 9):
    if i == 4: continue

    pdf = pdfs + '_' + str(i)

    page_num = 0

    doc = fitz.open(pdf + '.pdf')
    
    while True:
        try:
            print(pdf, 'opening page: ', page_num)
            page = doc.load_page(page_num)
            pixmap = page.get_pixmap(dpi=500)
            img = pixmap.tobytes()
            with open(pdf + '_' + str(page_num + 1) + '.png', 'wb') as f:
                f.write(img)
        except Exception as e:
            print(e)
            break

        page_num += 1
