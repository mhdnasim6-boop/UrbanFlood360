### How the numbers are produced

**1. Input data.** A Sentinel-1 radar pair (relative orbit 103, ascending): 19 October 2024 before the flood and
31 October 2024 after it. Six bands at 10 m resolution: VV and VH backscatter before and after the flood, and the
change between them in dB. Radar sees through cloud, which is why it is used for flood mapping.

**2. Two flood maps from the same U-Net.** The radar image was taken two days after the flood peak. By then most
streets in the towns held mud and debris, not water. Copernicus EMSR773 therefore maps two kinds of flood evidence:
*Flooded area* (standing water, mostly fields and the Albufera rice paddies) and *Flood trace* (receded flood,
covering the town centres). Two models are trained with the same code:

| Map | Training label | Used for |
|---|---|---|
| **Flood-affected area** | Flooded area + Flood trace | building damage assessment |
| **Standing water (31 Oct)** | Flooded area only | water still present on 31 Oct |

A standing-water map alone would miss most affected buildings, because only 0.2 % of building pixels fall inside
Copernicus's *Flooded area* polygons.

**3. The U-Net** (`models/unet_v2/`): a 4-level encoder–decoder with skip connections and batch normalisation,
about 7.8 M parameters, written for this project. It takes the 6 radar bands and outputs a probability for every
10 m pixel.

* *Labels:* areas outside the Copernicus image footprint and outside the six towns are unlabelled and do not
  count in training.
* *Training:* random 64 × 64 windows (640 m × 640 m) drawn from the training area of a 4 × 4 block split, kept
  more than 150 m away from the validation and test blocks. Loss = weighted binary cross-entropy + Dice. The saved
  weights are a moving average of the training weights. Each target was trained with and without rotation/flip
  augmentation, and the better run on the validation block was kept.
* *Model selection:* the checkpoint and the decision threshold (0.3–0.7) with the best Dice on the validation
  block. The test block is evaluated only once, at the end.
* *Inference:* overlapping 128 × 128 tiles, blended so no seams show, cover the whole area.

**4. Building impact.** For each of the 12,767 Catastro buildings, the flood-affected map is read over the
footprint plus a 10 m ring. At 10 m resolution the radar sees the streets and yards around a building, not its
interior.

| Class | Rule (share of footprint + ring classified as flood-affected) |
|---|---|
| Affected | ≥ 50 % |
| Possibly affected | 10 – 50 % |
| Not affected | < 10 % |
| No data | no radar pixel available |

Dwellings, floor area and use come from Catastro. **Residents are estimated** as dwellings × 2.5 people per
dwelling. This is an assumption, not census data; change it with
`scripts/assess_building_damage.py --persons-per-dwelling`.

**5. Check against Copernicus.** The EMSR773 grading product marks damaged buildings by photo-interpretation of
very-high-resolution optical images. Each EMSR point is matched to its Catastro building, and the share that this
method flags is reported in *Model performance*.

### Limitations

* **One event, small dataset.** The models learn from one flood in one area, so scores will be lower on other
  cities or events.
* **Validation scores are optimistic**, because the validation block was used to choose the checkpoint and
  threshold. The test block was not used for any choice. For standing water the test block contains very little
  water (113 pixels), so that Dice is unstable.
* **Labels exist for only ~73 % of the area**, the part imaged by Copernicus. Predictions outside it, including
  the eastern rice paddies, cannot be checked against the reference.
* **"Flood trace" is a visual interpretation** of mud and debris on optical images. The radar does not see mud
  directly; the model learns the changes that come with it (debris, wet surfaces, altered street geometry).
* **Urban radar is hard.** Buildings cause shadow, layover and double-bounce, and 10 m pixels mix buildings
  and streets.
* **"Affected" means flood evidence around the building**, not confirmed structural damage. Water depth,
  ground-floor use and basements are not modelled.
* **The Copernicus check is not independent.** EMSR773 flood polygons were also the training labels. EMSR lists only
  damaged buildings, so the check measures detection rate, not false alarms.
