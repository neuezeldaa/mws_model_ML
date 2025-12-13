# MWS Model ML

Коротко про проект:
На вход поступает распаршенный json в виде .csv таблицы 

Признаки должны быть в правильном порядке:
>RuleID
>
>Commit
>
>File
>
>StartLine
>
>EndLine
>
>StartColumn
>
>EndColumn
>
>Match
>
>Secret
>
>Date
>
>Message
>
>Author
>
>Email
>
>Fingerprint
>
>IsRealLeak

!!!Иначе модель работать не будет





На выходе модель отдает .csv файл, содержащий Secret,RuleID,File,IsRealLeak_true,CB_PredictedIsRealLeak,CB_Prob_RealLeak,CB_Prob_FalsePositive,CB_Confidence. 
Выход еще будет меняться, пока что это было тестово