Attribute VB_Name = "StripAccentModule"
Function StripAccent(cellString As String) As String
    Dim i As Long
    Dim A As String, B As String
    
    Const AccChars = "ְֱֲֳִֵַָֹÊֻּֽ־ֿ׀ׁׂ׃װױײ״ÙÚÛÜÝאבגדהוחטיךכלםמןנסעףפץצרשתûüÿ‎"
    Const RegChars = "SzYAAAAAACEEEEIIIIDNOOOOOOUUUUYaaaaaaceeeeiiiidnoooooouuuuyy"
    
    For i = 1 To Len(AccChars)
        A = Mid(AccChars, i, 1)
        B = Mid(RegChars, i, 1)
        cellString = Replace(cellString, A, B)
    Next
    
    StripAccent = cellString
End Function

