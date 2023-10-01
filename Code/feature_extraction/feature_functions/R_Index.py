
'''
Pressure Ratio - Single Foot (Ratio Index)
Ratio of the average heel pressure and average toe pressure on one foot.
This means that if the patient applies even pressure, the ratio should be around 1.
If the patient applies more pressure on the heel with respect to the toe, then the ratio is > 1.
If the patient applies less pressure on the heel with respect to the toe, then the ratio is < 1.
'''
def single_pressure_ratio(dn_complete, side='left'):
    return dn_complete.loc[:, 'avg ' + side + ' heel pressure'].mean() / dn_complete.loc[:, 'avg ' + side + ' toe pressure'].mean()


'''
Pressure Ratio - Both Feet
Ratio of the average heel pressure and average toe pressure on both feet.
First, it computes the ratio of the two single_pressure_ratio to see if the patient applies the same pressure on both feet.
Then, it computes the ratio of the heel and toe pressures on both feet, to see if the patient applies the same heel pressure
and toe pressure on both feet.

This means that if the patient applies even pressure, the ratio should be around 1.

For the single ratio:
If the patient applies more pressure on the left foot compared to the right foot, then the ratio is > 1.
If the patient applies less pressure on the left foot compared to the right foot, then the ratio is < 1.

For the double ratio:
If the patient applies more pressure on the left toe/heel compared to the right toe/ heel, then the ratio is > 1.
If the patient applies less pressure on the  left toe/heel compared to the right toe/ heel, then the ratio is < 1
'''
def double_pressure_ratio(dn_complete):
    single_ratio = single_pressure_ratio(dn_complete) / single_pressure_ratio(dn_complete, side='right')

    double_toe_ratio = dn_complete.loc[:, 'avg left toe pressure'].mean() / dn_complete.loc[:, 'avg right toe pressure'].mean()
    double_heel_ratio = dn_complete.loc[:, 'avg left heel pressure'].mean() / dn_complete.loc[:, 'avg right heel pressure'].mean()

    return single_ratio, double_toe_ratio, double_heel_ratio