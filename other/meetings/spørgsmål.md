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
    1. Burde jeg gøre brug af en GRU istedet for en LSTM?
    2. Many-to-many eller many-to-one RNN?

*   Ubesvaret spørgsmål:
    1. Ved "Related Work", skal jeg kun fokusere på pose estimation eller også andet object tracking?
    2. Nogle BRACE-videos indeholder flere personer, så det kan måske være svært for modellen at vide hvem den skal predicte på.
    Burde jeg bruge bbox til at centrere videoen omkring personen som skal predictes på?
    3. Hvad med teacher forcing?
