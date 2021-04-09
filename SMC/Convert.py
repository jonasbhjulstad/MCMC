import csv

with open(r'C:\Users\Jonas\Downloads\IMMA-MarineObs_icoads3.0.1-Prelim_d202101_c20210210102002.dat\IMMA-MarineObs_icoads3.0.1-Prelim_d202101_c20210210102002.dat') as infile, open(r'C:\Users\Jonas\Downloads\IMMA-MarineObs_icoads3.0.1-Prelim_d202101_c20210210102002.dat\IMMA.csv', "w") as outfile:
    csv_writer = csv.writer(outfile)
    prev = ''
    csv_writer.writerow(['ID', 'PARENT_ID'])
    for line in infile.read().splitlines():
        csv_writer.writerow([line, prev])
        prev = line