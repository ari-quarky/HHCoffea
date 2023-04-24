import os

#print("|  *m_LQ [GeV]*  |  * &sigma; (NLO) [pb] *  |  * &delta;_stat *  |  * &delta;_scaleUp (%)*  |  * &delta;_scaleDown (%)*  |  * &delta;_PDFUp (%)*  |  * &delta;_PDFDown (%)*  |")

xsecs=[]
pdfErrs=[]
scaleErrs=[]
xsec=scaleErr=pdfErr=statErr=''

resultsDir = 'LQLQ_Leptokvark_pair'
dirs = [ff.replace('\n','').split('run_')[-1] for ff in os.popen('ls '+resultsDir+"/Events | grep \"run_\"").readlines()]

print(dirs)
dirs.sort()

for dir in dirs:
  if dir<10:
    dir='0'+str(dir)
  x=str(dir)
  filename = resultsDir+'/Events/run_'+x+'/summary.txt'
  filenameMass = resultsDir+'/Events/run_'+x+'/run_'+x+'_tag_1_banner.txt'
  #print filename
  masslines = [ff.replace('\n','') for ff in os.popen('cat '+filenameMass).readlines()]
  for line in masslines:
    if '# mlq' in line:
      mass = int(float(line.split('# mlq')[0].split('9000002')[-1]))
      #print mass
  lines = [ff.replace('\n','') for ff in os.popen('cat '+filename).readlines()]
  #print lines
  for y in range(len(lines)):
    if 'Total cross section' in lines[y]:
        xsec=lines[y]
        xsecs.append(lines[y])
    if 'Scale variation' in lines[y]:
        scaleErr=lines[y+2]
        scaleErrs.append(lines[y+2])
    if 'PDF variation' in lines[y]:
        pdfErr=lines[y+2]
        pdfErrs.append(lines[y+2])
  xsecFinal = xsec.split(':')[-1].split('pb')[0].split('+-')[0].replace(' ','')
  statErr = xsec.split(':')[-1].split('pb')[0].split('+-')[-1].replace(' ','')
  scaleErrUp = scaleErr.split('pb')[-1].split('-')[0].replace('+','').replace('%','').replace(' ','')
  scaleErrDown = scaleErr.split('pb')[-1].split('-')[-1].replace('%','').replace(' ','')
  #print pdfErr
  pdfErrUp = pdfErr.split('pb')[-1].split('-')[0].replace('+','').replace('%','').replace(' ','')
  pdfErrDown = pdfErr.split('pb')[-1].split('-')[-1].replace('%','').replace(' ','')
  #print pdfErrUp,pdfErrDown
  print('| ',mass,' | ',xsecFinal,' | ',statErr,' | ',scaleErrUp,' | ',scaleErrDown,' | ',pdfErrUp,' | ',pdfErrDown,' |')
xsecsFinal = [f.split(':')[-1].split('pb')[0].split('+-')[0] for f in xsecs]
statErrs = [f.split(':')[-1].split('pb')[0].split('+-')[-1] for f in xsecs]
print(xsecsFinal)
#print statErr
