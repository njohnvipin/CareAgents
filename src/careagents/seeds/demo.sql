-- Fictional demo data. Mirrors careagents/data/memory.py.
INSERT INTO plans VALUES ('HSG-2026', 'HealthSure Gold 2026', 2026, 500, 20, 5000) ON CONFLICT DO NOTHING;

INSERT INTO members VALUES
  ('SARAH-1001', 'HSG-2026', true, '2026-01-01', NULL),
  ('JOHN-2002',  'HSG-2026', true, '2026-01-01', NULL)
ON CONFLICT DO NOTHING;

INSERT INTO accumulators VALUES
  ('SARAH-1001', 2026, 200, 200),
  ('JOHN-2002',  2026, 500, 900)
ON CONFLICT DO NOTHING;

INSERT INTO services VALUES
  ('73721', 'MRI knee (lower extremity joint) without contrast',
   ARRAY['mri','knee mri','mri knee','magnetic resonance imaging'], 'advanced_imaging', true, 1000),
  ('99204', 'New patient office visit (orthopedic consultation)',
   ARRAY['consultation','office visit','orthopedic consult'], 'office_visit', false, 200),
  ('97110', 'Physical therapy, therapeutic exercise',
   ARRAY['physical therapy','pt','physio'], 'therapy', false, 120)
ON CONFLICT DO NOTHING;

INSERT INTO claims VALUES
  ('CLM-7781', 'SARAH-1001', '99204', '2026-01-10', 'Lakeside Orthopedics', 200, 'approved',
   NULL, 'Applied to deductible', 200, 0),
  ('CLM-8892', 'SARAH-1001', '73721', '2026-02-05', 'North Dallas Imaging Center', 1000, 'denied',
   'PA_MISSING', 'Prior authorization not obtained before service date', 0, 0),
  ('CLM-9001', 'JOHN-2002', '97110', '2026-03-02', 'Uptown Physio', 120, 'approved', NULL, NULL, 24, 96)
ON CONFLICT DO NOTHING;

INSERT INTO prior_authorizations VALUES
  ('PA-5501', 'JOHN-2002', '73721', 'approved', '2026-03-10T00:00:00Z', '2026-03-12T00:00:00Z', '2027-03-12')
ON CONFLICT DO NOTHING;
