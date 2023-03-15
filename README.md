# Arch_Class
Architecture Classification for Non Traditional Builds

New repo for classification of Non Traditional Builds via Street View data 

Workflow 
- 1. Ingest Freedom of Information (FOI) requests from UK local authorities on locations of non traditional builds 
- 2. Process FOI to create clean address -> non trad mapping 
- 3. Geo code addresses in Google 
- 4. Download Street view imagery (SVI) for addresses from geocode 
- 5. Use InceptionV3 trained on Places365 data to filter facades only from SVI
- 6. Train DCNN on facades to identify non traditional builds 

Github in progress 
