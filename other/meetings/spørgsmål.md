-   2/2:

    1. Data spørgsmål:
        1. Har også fundet andre datasæt - hvad med pretæning på disse og så finetune på ClimbAlong data?
        2. Måske kombinere flere datasæt?
    2. Model spørgsmål:
        1. Skal vi bare scratche den allerede implementerede Mask R-CNN eller skal vi udvide den?
        2. Skal jeg også inkorporere noget temporal-information i tracking af grips eller skal vi bare fokusere på pose estimation?
        3. Forskellige variationer af den samme model eller skal vi eksperimentere med forskellige modeller?
        4. Single- eller multi-person tracking?
        5. Nogle modeller gør brug af en 3D convolution istedet for en RNN - burde jeg også prøve dette eller skal vi bare holde os til RNN?
    3. Generelle spørgsmål:
        1. Hvor meget ML teori?
        2. Foreslå fremgangsmåde:
            1. Implementer temporal model
            2. Træn temporal model på et public dataset - kun på skeleterne (+ noise?)
            3. Finetune på ClimbAlong datasæt (+ noise?)
            4. Finetune hele R-CNN + temporal model på ClimbAlong?

-   9/2:

    1. Dataset:

        1. BRACE ser god ud synes jeg - nogle frames er dog ikke helt korrekt annoteret. Videoerne har dog nogle udfordringer (wide panning, long zooming, aerial views, abrupt shot changes and lighting)
        2. Data augmentation på BRACE?
        3. Data preprocessing af BRACE?
            1. Samme aspect ratio som ClimbAlong?
            2. Trække mean CLimbAlong trænings RGB fra?
        4. Brace har nogle keypoints som måske ikke så er nødvendige for os (øjne og øre) - skal vi bare fjerne disse?
        5. Brace har nogle "rejected" frames - hvordan handler vi disse?

    2. Model:

        1. Mange modeller bruger ikke ResNet som backbone, men istedet HRNet - skal jeg også prøve dette?
        2. Unipose-LSTM ser stadig spændende ud
        3. DeciWatch ser også spændende ud (især fordi den er efficient)
        4. Skal jeg prøve både at lave en/to LSTM-modeller (Unipose) og en Transformer-model (DeciWatch)?

    3. Andre spørgsmål:
        1. Skal jeg køre med lavere precision (eks. 16-bit)?
        2. Ved model som kun processerer poses, skal vi så tilføje noget noise til input-pose, sådan så de kommer til at ligne outputet af R-CNN mere?
        3. Bezier interpolation som baseline?

- 17/2:
    1. Dataset:
        1. Hvilken preprocessing?
        2. BRACE er landscape - skal jeg prøve at lave det om til portrait mode?
        3. Skal BRACE-videoerne gemmes som tensor eller mp4?

- 23/2:
    1. Dataset:
        1. COCO eller 25 keypoints?
        2. Nogle BRACE-videos indeholder flere personer. Burde jeg bruge bbox til at centrere videoen omkring personen som skal predictes på?

    2. Model:
        1. Burde jeg gøre brug af en GRU istedet for en LSTM?
        2. Many-to-many eller many-to-one RNN?
        3. Hvad gør man med batches når samples har forskellig størrelse?
        4. Skal jeg selv implementere Mask R-CNN eller bruge den fa ClimbAlong?
- 8/3:
    1. Dataset: 
        1. Nogle frames har jeg altså bare gættet på placeringen af keypoints (især for fingrene)
        2. For de allerede pre-annoterede billeder, er højre og venstre hånd nogle gang byttet om.
    
    2. Model:
        1. Lav en liste over de modeller som skal implementeres.
            * Baseline:
                * Window-size

            * Transformer-based:
                * Antal transformer blocvks
                * Embedding dimension
                * Frame sample-rate

            * RNN-based:
                * RNN-type
                * Placering
                * Bidirectional?
                
        2. Kombination af info fra begge directions i bidirectional rnn?

    2. Kommende uge:
        1. Forbered alt pre-training, sådan så den kan pre-trænes til næste meeting

    3. Report:
        1. Ved "Related Work", skal jeg kun fokusere på pose estimation eller også andet object tracking?

- 28/3:
    1. Model:
        1. Hvilken loss-function?
        2. Hvilken evaluation metric?

- 13/4:
    1. Hvorfor har jeg haft problemer med clusteret?
    2. Miskommunikation imellem mig og ClimbAlong?
    3. Fejl i preprocessing?
    4. Fjerner jeg gradient på den rigtige måde?
    5. Skal jeg beskrive mask RCNN?

- 25/4:
    1. Jeg har ændres på shifting-std.
    2. Jeg har rettet PCK
    3. Er det på tide at jeg finetuner?
    4. Er der noget jeg skal gøre anderledes når jeg finetuner?

- 3/5:
    1. Der har næsten været én lang kø på clusteret, så har ikke fået trænet flere forskellige noise-std
    2. Jeg blev informeret om heatmaps var i størrelse 50 x 50, men dem jeg har fået er i 56 x 56
    3. Andrea har kun sendt 25 kpt annoteringer
    4. Mask RCNN tager ikke højde for aspect ratio som jeg ellers gør.
    5. Sigmoid som sidste lag lader ikke til at forbedre dataen
    6. Unipose2 ser ud til at performe lige så godt som Unipose
    7. Hvordan jeg laver gt-heatmaps til finetuning
    8. Hvad skal jeg gøre når gt-kp ligger uden for pred-bbox?
    9. Hvad gør jeg når pred-bbox ikke findes (eks. når personen er halvt ude af billedet)?
        1. Bruge linear interpolation til at lave bbox?
        2. Dette sker dog kun til sidst i en video
    10. Preprocessing af CA_data (heatmap[heatmap < 0] = 0 og heatmap = heatmap/heatmap.sum())
    11. Hvorfor ikke bruge predicted keypoint til at lave finetuning heatmap?
    12. Hvad skal jeg tage højde for når jeg fine-tuner?
    13. Data-split til finetuning? (evt. cross-validation?)
    14. Hvordan vælger jeg hvilke modeller der skal finetunes?
    15. Flytning af forsvarsdato
    16. Handler jeg std forkert?

- 10/5:
    1. Data augmentation når jeg finetuner?
    2. Introducerer jeg noget bias når jeg laver val/test-split?
    3. Skal jeg også gå i detaljer med Mask RCNN?
    4. PCK for de forskellige dataset?
    5. Det ligner lidt, at pretraining måske ikke har hjulpet så meget?
    6. Handler jeg std forkert?
    7. Flytning af forsvarsdato?
    8. Er der noget bias, i og med at Mask RCNN kan være trænet på min valideringsdata
    9. Handling af koordinator når keypoint mangler?
    10. Navn på teori der siger der er en sammenhæng imellem datastørrelse og antal parametrer?
    11. Testing accuracies:
        1. Lidt støj:
            1. Baseline: 0.993 - 0.994
            2. Transformer: 0.991
            3. ConvLSTM (concat): 0.997 - 0.998
            4. ConvLSTM (sum): 0.995 - 0.997
        2. Meget støj:
            1. Baseline: 0.994 - 0.995
            2. Transformer: 0.942 - 0.968
            3. ConvLSTM (concat): 0.995 - 0.997
            4. ConvLSTM (sum): 0.993 - 0.997
        3. Generelt:
            1. ConvLSTM giver de bedste resultater - dog kun lidt bedre end baseline
            2. Det hjalp faktisk at concatenate branches istedet for at summere
            3. Transformer giver altid de dårligste resultater
            4. Støjen fra pretraining har stor effekt på transformeren (måske skyldet dens mange vægte?)