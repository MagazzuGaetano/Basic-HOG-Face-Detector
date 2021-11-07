# Face-Detector
A simple face detector svm + hog sliding window approach
Non Maximum Suppression Per ridurre le bounding box

# Dati Utilizzati:
- faces: LWZ in the wild (~13000 samples)
- not faces: caltech 256 dataset (~26000 samples)
- not faces 2: natural images 10 classes kaggle
- not faces 3: dtd texture dataset


from the caltech 256 dataset the following classes was removed:

religious: buddha, jesuschrist
human: faces_easy, human_skeleton, people
cartoon: superman, homer-simpson, cartman

- limitatezza multi scala va tunato meglio
- performance non bellissime (tempo calcolo)
- qualità immagine in input noon deve essere troppo rumorosa o smussata
- non pensato per riconoscere faccie girate di profilo o di spalle
- non pensato per risolvere problemi di occlusione (occhiali, mascherina, ...)

# condizioni ideali:
- faccia intera frontale, senza occlusioni
- un solo soggetto ad una scala abbastanza grande ma non troppo
- se sono più soggetti ad una certa distanza tra di loro
- (le faccie piccole ed attaccate tra di loro sono più difficili)
