import pandas as pd


def aggregate_by_week_and_state(sales):
    return sales.groupby(['State', 'year', 'week'], as_index=False).sum()


def combine_sales_with_loyalty(sales, loyalty, stores):
    sales['week'] = sales['Date'].dt.week
    sales['year'] = sales['Date'].dt.year
    loyalty['week'] = loyalty['CalendarWeekEndingDate'].dt.week
    loyalty['year'] = loyalty['CalendarWeekEndingDate'].dt.year

    sales_with_state = pd.merge(
        sales,
        stores,
        how='left',
        on='StoreId'
    )

    sales_with_state = aggregate_by_week_and_state(sales_with_state)

    sales_with_loyalty = pd.merge(
        sales_with_state,
        loyalty,
        how='left',
        on=['State', 'year', 'week']
    )

    sales_with_loyalty = sales_with_loyalty.drop(columns=['StoreId'])

    return sales_with_loyalty
