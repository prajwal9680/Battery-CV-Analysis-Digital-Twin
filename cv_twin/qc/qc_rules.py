def qc_status(D, rmse_percent):
    """
    Return PASS/WARN/FAIL with reasons based on simple heuristics.
    """
    issues = []
    if not (1e-11 <= D <= 1e-9):
        issues.append('D out of range')
    if rmse_percent > 8.0:
        issues.append('RMSE high')
    if not issues:
        return 'PASS', issues
    return ('WARN' if len(issues) == 1 else 'FAIL'), issues
