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